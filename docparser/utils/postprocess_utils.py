import logging
import os
from xml.etree import ElementTree

import lxml.etree as et
from PIL import Image
from glob import glob
from xml.dom import minidom

logger = logging.getLogger(__name__)


def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def second_bbox_overlaps_with_first_bbox_by_percentage_xyformat(b1, b2, percentage=0.5):
    bbox2_area = (b2[3] - b2[1]) * (b2[2] - b2[0])
    bbox1_area = (b1[3] - b1[1]) * (b1[2] - b1[0])
    if bbox2_area == 0 or bbox1_area == 0:
        return False
    [x0, y0, x1, y1] = b2
    [struct_x0, struct_y0, struct_x1, struct_y1] = b1
    if x1 <= struct_x0:
        return False
    elif x0 >= struct_x1:
        return False
    overlap_x0 = max(x0, struct_x0)
    overlap_x1 = min(x1, struct_x1)
    overlap_width = overlap_x1 - overlap_x0

    if y1 <= struct_y0:
        overlap = 0
        return False
    elif y0 >= struct_y1:
        overlap = 0
        return False
    overlap_y0 = max(y0, struct_y0)
    overlap_y1 = min(y1, struct_y1)
    overlap_height = overlap_y1 - overlap_y0

    overlap_area = overlap_width * overlap_height
    if float(overlap_area) / bbox2_area > percentage:
        return True
    else:
        return False


def match_text_elements_with_cell_bbox(cell_bbox, text_elems_for_page):
    matched_contents = []
    # sort text elems:
    # low y0 value: 'lower' on page. note the minus before the first lambda expression to sort by y value reversed
    sorted_text_elems_for_page = sorted(text_elems_for_page, key=lambda x: (
        -(x['flippedbbox'][1] + (x['flippedbbox'][3] - x['flippedbbox'][1]) / 2.0),
        (x['flippedbbox'][0] + (x['flippedbbox'][2] - x['flippedbbox'][0]) / 2.0)))
    for text_elem in sorted_text_elems_for_page:
        elem_bbox = text_elem['flippedbbox']
        elem_content = text_elem['content']
        if second_bbox_overlaps_with_first_bbox_by_percentage_xyformat(cell_bbox, elem_bbox, percentage=0.5):
            matched_contents.append(elem_content.strip())
    return matched_contents


def retrieve_text_elements_for_page(doc_name, detections_per_doc, elems_folder):
    elems_file_path = os.path.join(elems_folder, doc_name + '-elem.xml')
    root = et.parse(elems_file_path)
    pages = root.findall('page')
    text_elems_per_page = dict()
    for page in pages:
        page_text_elems = []
        page_num = page.get('num')
        text_elements = page.findall('element')
        for text_element in text_elements:
            text_content_el = text_element.find('content')
            bbox_el = text_element.find('bounding-box')
            text_content = text_content_el.text
            x0 = int(bbox_el.get('x1'))
            y0 = int(bbox_el.get('y1'))
            x1 = int(bbox_el.get('x2'))
            y1 = int(bbox_el.get('y2'))
            # note: we first have to subtract height to make this valid
            flippedbbox = [x0, y0, x1, y1]
            page_text_elems.append({'flippedbbox': flippedbbox, 'content': text_content})
        text_elems_per_page[int(page_num)] = page_text_elems

    return text_elems_per_page


def generate_xmls_per_doc(doc_name, detections_per_doc, output_xml_dir, original_doc_images, text_elems_per_page):
    if not os.path.exists(output_xml_dir):
        os.mkdir(output_xml_dir)

    new_xml_file = os.path.join(output_xml_dir, doc_name + '-str-gt_output.xml')
    xml_root = et.Element('document')
    xml_root.set('filename', doc_name)

    for page_nr in detections_per_doc[doc_name].keys():
        sorted_table_nrs = sorted(list(detections_per_doc[doc_name][page_nr].keys()))

        img = Image.open(original_doc_images[page_nr])
        all_page_anns = []
        width, height = img.size

        for table_nr in sorted_table_nrs:
            sorted_region_nrs = sorted(list(detections_per_doc[doc_name][page_nr][table_nr]), key=lambda x: int(x))
            for true_region_count, region_nr in enumerate(sorted_region_nrs):
                current_region_annotations = detections_per_doc[doc_name][page_nr][table_nr][region_nr]['annotations']
                all_page_anns += current_region_annotations

        for table_nr in sorted_table_nrs:

            table_node = et.SubElement(xml_root, 'table')
            table_node.set('id', str(table_nr))

            previous_rows = None
            previous_cols = None
            sorted_region_nrs = sorted(list(detections_per_doc[doc_name][page_nr][table_nr]), key=lambda x: int(x))
            for true_region_count, region_nr in enumerate(sorted_region_nrs):
                row_increment = 0
                col_increment = 0

                current_region_annotations = detections_per_doc[doc_name][page_nr][table_nr][region_nr]['annotations']
                num_rows = len([ann for ann in current_region_annotations if ann['category'] == 'table_row'])
                num_cols = len([ann for ann in current_region_annotations if ann['category'] == 'table_col'])
                if true_region_count >= 1 and num_cols == previous_cols:
                    row_increment = previous_rows
                    logger.debug('same col count in region {}, using row increment: {}'.format(num_cols, row_increment))
                elif true_region_count >= 1 and num_rows == previous_rows:
                    col_increment = previous_cols
                    logger.debug('same col count in region {}, using row increment: {}'.format(num_cols, row_increment))
                previous_rows = num_rows
                previous_cols = num_cols

                region_node = et.SubElement(table_node, 'region')
                region_node.set('id', str(region_nr))
                region_node.set('page', str(page_nr + 1))  # seems to be counted from 1 in icdar
                region_node.set('row_increment', str(row_increment))
                region_node.set('col_increment', str(col_increment))
                for cell_nr, ann in enumerate(
                        [ann for ann in current_region_annotations if ann['category'] == 'table_cell']):
                    if ann['category'] == 'table_cell':
                        [start_row, end_row] = ann['row_range']
                        [start_col, end_col] = ann['col_range']
                        cell_node = et.SubElement(region_node, 'cell')
                        cell_node.set('id', str(cell_nr))
                        cell_node.set('start-row', str(start_row))
                        cell_node.set('start-col', str(start_col))
                        if end_row != start_row:
                            cell_node.set('end-row', str(end_row))
                        if end_col != start_col:
                            cell_node.set('end-col', str(end_col))
                        [x0, y0, w, h] = ann['bbox']
                        x1 = x0 + w
                        y1 = y0 + h

                        tmp = y0
                        y0 = height - y1
                        y1 = height - tmp
                        # NOTE: y1 and y0 are 'flipped'. we have to get the 72dpi image and subtract the values from its height

                        bbox_node = et.SubElement(cell_node, 'bounding-box')
                        bbox_node.set('x1', str(x0))
                        bbox_node.set('y1', str(y0))
                        bbox_node.set('x2', str(x1))
                        bbox_node.set('y2', str(y1))

                        matched_contents = None
                        matched_contents = match_text_elements_with_cell_bbox([x0, y0, x1, y1],
                                                                              text_elems_per_page[page_nr + 1])
                        if len(matched_contents) > 0:
                            new_contents = ' '.join(matched_contents)
                            ann['content'] = new_contents
                        else:
                            ann['content'] = ''
                        bbox_node = et.SubElement(cell_node, 'content')
                        bbox_node.text = str(ann['content'])

    xml_file_contents = prettify(xml_root)
    with open(new_xml_file, 'w') as out_file:
        out_file.write(xml_file_contents)
    logger.debug('created xml file at {}'.format(new_xml_file))


def get_icdar_info_from_filename(file_name):
    doc_name = file_name.split('_p')[0]
    page_nr = int(file_name.split('_p')[1].split('_')[0])
    table_nr = int(file_name.split('_table')[1].split('_')[0])
    region_nr = int(file_name.split('_reg')[1].split('_')[0])
    counter_nr = int(file_name.split('_nr')[1].split('-')[0])
    return doc_name, page_nr, table_nr, region_nr, counter_nr


def get_detections_from_file(detection_file_path, page_nr=0, header=True):
    with open(detection_file_path, 'r') as in_file:
        if header is True:
            header_line = in_file.readline()
            header_split = header_line.split(';')
            assert len(header_split) > 0
            height = header_split[0].split(':')[-1]
            width = header_split[1].split(':')[-1]
            img_size = (width, height)

        detection_lines = in_file.readlines()
        annotations = []
        ann_id_counter = 0
        for detection_line in detection_lines:
            if len(detection_line.split()) == 7:
                [original_ann_id, class_name, pred_score, x0, y0, x1, y1] = detection_line.split()
            elif len(detection_line.split()) == 6:
                [original_ann_id, class_name, x0, y0, x1, y1] = detection_line.split()
                pred_score = None
            else:
                raise NotImplementedError

            bbox = [float(x0), float(y0), float(x1) - float(x0), float(y1) - float(y0)]
            annotation = {'id': ann_id_counter, 'category': class_name, 'bbox': bbox, 'score': pred_score,
                          'page': page_nr}
            ann_id_counter += 1
            annotations.append(annotation)
    return annotations, img_size


def get_detections_for_directory(fullsize_predictions):
    detections_per_doc = dict()
    for detection_file in glob(fullsize_predictions + '/*.txt'):
        file_name = os.path.relpath(detection_file, fullsize_predictions)

        doc_name, page_nr, table_nr, region_nr, counter_nr = get_icdar_info_from_filename(file_name)
        if doc_name not in detections_per_doc:
            detections_per_doc[doc_name] = dict()
        if page_nr not in detections_per_doc[doc_name]:
            detections_per_doc[doc_name][page_nr] = dict()
        if table_nr not in detections_per_doc[doc_name][page_nr]:
            detections_per_doc[doc_name][page_nr][table_nr] = dict()
        detections_per_doc[doc_name][page_nr][table_nr][region_nr] = {'file_name': file_name}
        annotations, img_size = get_detections_from_file(detection_file, page_nr=page_nr)
        detections_per_doc[doc_name][page_nr][table_nr][region_nr]['annotations'] = annotations
        detections_per_doc[doc_name][page_nr][table_nr][region_nr]['img_size'] = img_size
    return detections_per_doc


def get_crop_region(crop_data_folder, doc_name, page_nr, table_nr, region_nr, ann_version):
    matching_string = '{}_p{}_table{}_reg{}'.format(doc_name, page_nr, table_nr, region_nr)
    crop_info_files = glob(
        crop_data_folder + '/{}*/{}*cropinfo-{}.txt'.format(matching_string, matching_string, ann_version),
        recursive=True)
    assert len(crop_info_files) == 1
    crop_info_file = crop_info_files[0]
    with open(crop_info_file, 'r') as in_file:
        [header, crop_info] = in_file.readlines()
    [crop_x0, crop_y0, crop_x1, crop_y1] = [float(x) for x in crop_info.split(', ')]
    return [crop_x0, crop_y0, crop_x1, crop_y1]
