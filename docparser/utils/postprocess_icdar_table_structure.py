import logging
import os

import copy

from docparser.utils.postprocess_table_structure import adjust_bboxes_to_true_page_coordinates, \
    resolve_direct_nesting_of_rows_and_columns, assign_numbers_to_rows_and_cols, adjust_and_move_row_and_column_borders, \
    limit_all_annotations_to_crop_region, create_all_cells_and_assign_coordinates
from docparser.utils.postprocess_utils import get_detections_for_directory, generate_xmls_per_doc, \
    retrieve_text_elements_for_page, get_crop_region

logger = logging.getLogger(__name__)


def process_detections_for_icdar_data(detections_dir, output_xml_dir, original_data_folder, crop_data_folder,
                                      ann_version, elems_folder):
    detections_per_doc = get_detections_for_directory(detections_dir)

    for doc_name in detections_per_doc.keys():
        doc_subdir = os.path.join(original_data_folder, doc_name)
        original_doc_images = {int(x.rsplit('-')[-1].split('.png')[0]): os.path.join(doc_subdir, x) for x in
                               os.listdir(doc_subdir) if x.endswith('.png') and not 'char' in x}
        logger.debug('current doc: {}'.format(doc_name))

        for page_nr in detections_per_doc[doc_name].keys():
            for table_nr in detections_per_doc[doc_name][page_nr].keys():
                sorted_region_nrs = sorted(list(detections_per_doc[doc_name][page_nr][table_nr]), key=lambda x: int(x))
                for region_nr in sorted_region_nrs:
                    [crop_x0, crop_y0, crop_x1, crop_y1] = get_crop_region(crop_data_folder, doc_name, page_nr,
                                                                           table_nr, region_nr, ann_version)
                    crop_bbox = [crop_x0, crop_y0, crop_x1 - crop_x0, crop_y1 - crop_y0]

                    current_region_annotations = detections_per_doc[doc_name][page_nr][table_nr][region_nr][
                        'annotations']

                    adjust_bboxes_to_true_page_coordinates(current_region_annotations, crop_x0, crop_y0)
                    current_region_annotations = resolve_direct_nesting_of_rows_and_columns(current_region_annotations)
                    num_rows, num_cols = assign_numbers_to_rows_and_cols(current_region_annotations)
                    structured_annotations = copy.deepcopy(current_region_annotations)
                    structured_annotations = adjust_and_move_row_and_column_borders(structured_annotations)
                    structured_annotations = limit_all_annotations_to_crop_region(structured_annotations, crop_bbox)
                    structured_annotations = create_all_cells_and_assign_coordinates(structured_annotations)
                    detections_per_doc[doc_name][page_nr][table_nr][region_nr]['annotations'] = structured_annotations

        text_elems_per_page = retrieve_text_elements_for_page(doc_name, detections_per_doc, elems_folder)
        generate_xmls_per_doc(doc_name, detections_per_doc, output_xml_dir, original_doc_images, text_elems_per_page)

    return detections_per_doc
