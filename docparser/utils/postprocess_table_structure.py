import logging
from collections import defaultdict

import copy

logger = logging.getLogger(__name__)


def get_row_overlaps(row_anns, cell_ann):
    row_overlaps = []
    for row_nr, row_ann in row_anns.items():
        if has_height_overlap_with_first_bbox_cocoformat(row_ann['bbox'], cell_ann['bbox'], percentage=0.5):
            row_overlaps.append(row_nr)
    return sorted(row_overlaps)


def get_col_overlaps(col_anns, cell_ann):
    col_overlaps = []
    for col_nr, col_ann in col_anns.items():
        if has_width_overlap_with_first_bbox_cocoformat(col_ann['bbox'], cell_ann['bbox'], percentage=0.5):
            col_overlaps.append(col_nr)
    return sorted(col_overlaps)


def adjust_and_move_row_and_column_borders(annotations):
    row_anns = dict()
    col_anns = dict()
    for ann in annotations:
        if ann['category'] == 'table_row':
            row_anns[ann['row_nr']] = ann
        elif ann['category'] == 'table_col':
            col_anns[ann['col_nr']] = ann

    # move shared column borders to respective midpoints
    col_nrs = sorted(list(col_anns.keys()))
    if len(col_nrs) > 1:
        for col_nr in col_nrs[:-1]:
            next_col_nr = col_nr + 1
            cur_col_x0, cur_col_y0, cur_col_w, cur_col_h = col_anns[col_nr]['bbox']
            cur_col_x1 = cur_col_x0 + cur_col_w
            cur_col_y1 = cur_col_y0 + cur_col_h
            next_col_x0, next_col_y0, next_col_w, next_col_h = col_anns[next_col_nr]['bbox']
            next_col_x1 = next_col_x0 + next_col_w
            next_col_y1 = next_col_y0 + next_col_h

            x_diff = next_col_x0 - cur_col_x1
            x_midpoint = cur_col_x1 + x_diff * 0.5
            cur_col_x1 = x_midpoint
            next_col_x0 = x_midpoint
            col_anns[col_nr]['bbox'] = [cur_col_x0, cur_col_y0, cur_col_x1 - cur_col_x0, cur_col_y1 - cur_col_y0]
            col_anns[next_col_nr]['bbox'] = [next_col_x0, next_col_y0, next_col_x1 - next_col_x0,
                                             next_col_y1 - next_col_y0]

    # move shared row borders to respective midpoints
    row_nrs = sorted(list(row_anns.keys()))
    if len(row_nrs) > 1:
        for row_nr in row_nrs[:-1]:
            next_row_nr = row_nr + 1
            cur_row_x0, cur_row_y0, cur_row_w, cur_row_h = row_anns[row_nr]['bbox']
            cur_row_x1 = cur_row_x0 + cur_row_w
            cur_row_y1 = cur_row_y0 + cur_row_h
            next_row_x0, next_row_y0, next_row_w, next_row_h = row_anns[next_row_nr]['bbox']
            next_row_x1 = next_row_x0 + next_row_w
            next_row_y1 = next_row_y0 + next_row_h

            y_diff = next_row_y0 - cur_row_y1
            y_midpoint = cur_row_y1 + y_diff * 0.5
            cur_row_y1 = y_midpoint
            next_row_y0 = y_midpoint
            row_anns[row_nr]['bbox'] = [cur_row_x0, cur_row_y0, cur_row_x1 - cur_row_x0, cur_row_y1 - cur_row_y0]
            row_anns[next_row_nr]['bbox'] = [next_row_x0, next_row_y0, next_row_x1 - next_row_x0,
                                             next_row_y1 - next_row_y0]

    # expand rows and columns such that their borders align on the outer border of the table
    tabular_max_x, tabular_max_y, tabular_min_x, tabular_min_y = -1, -1, 100000, 100000
    for rowcol in list(row_anns.values()) + list(col_anns.values()):
        x0, y0, w, h = rowcol['bbox']
        x1 = x0 + w
        y1 = y0 + h
        if x1 > tabular_max_x:
            tabular_max_x = x1
        if y1 > tabular_max_y:
            tabular_max_y = y1
        if x0 < tabular_min_x:
            tabular_min_x = x0
        if y0 < tabular_min_y:
            tabular_min_y = y0

    num_rows = len(row_anns)
    num_cols = len(col_anns)
    new_row_x0 = tabular_min_x
    new_row_w = tabular_max_x - tabular_min_x
    new_col_y0 = tabular_min_y
    new_col_h = tabular_max_y - tabular_min_y
    for row in row_anns.values():
        x0, y0, w, h = row['bbox']
        # Adjust left and right borders of all rows to the maximum outer border occurring in the table
        row['bbox'] = [new_row_x0, y0, new_row_w, h]
        # Adjust upper and lower borders of first and last row
        if row['row_nr'] == 0:
            new_h = row['bbox'][3] + row['bbox'][1] - tabular_min_y
            row['bbox'][1] = tabular_min_y
            row['bbox'][3] = new_h
        if row['row_nr'] == num_rows - 1:
            new_h = tabular_max_y - row['bbox'][1]
            row['bbox'][3] = new_h

    for col in col_anns.values():
        x0, y0, w, h = col['bbox']
        # Adjust upper and lower borders of all columns to the maximum outer border occurring in the table
        col['bbox'] = [x0, new_col_y0, w, new_col_h]
        # Adjust left and right borders of first and last column
        if col['col_nr'] == 0:
            new_w = col['bbox'][2] + col['bbox'][0] - tabular_min_x
            col['bbox'][0] = tabular_min_x
            col['bbox'][2] = new_w
        if col['col_nr'] == num_cols - 1:
            new_w = tabular_max_x - col['bbox'][0]
            col['bbox'][2] = new_w

    return annotations


def create_all_cells_and_assign_coordinates(annotations):
    detected_cells = [ann for ann in annotations if ann['category'] == 'table_cell']

    row_anns = dict()
    col_anns = dict()
    existing_grid_cells = set()
    for ann in annotations:
        if ann['category'] == 'table_row':
            row_anns[ann['row_nr']] = ann
        elif ann['category'] == 'table_col':
            col_anns[ann['col_nr']] = ann

    # Discard all detected non-multi cells
    normal_cells_delete_ids = set()
    for cell_ann in detected_cells:
        row_overlaps = get_row_overlaps(row_anns, cell_ann)
        col_overlaps = get_col_overlaps(col_anns, cell_ann)
        if len(row_overlaps) > 0 and len(col_overlaps) > 0 and (len(row_overlaps) > 1 or len(col_overlaps) > 1):
            logger.debug('multi cell:  rows: {}, cols: {}'.format(row_overlaps, col_overlaps))
            cell_ann['row_range'] = [row_overlaps[0], row_overlaps[-1]]
            cell_ann['col_range'] = [col_overlaps[0], col_overlaps[-1]]
            cell_ann['properties'] = "{}-{},{}-{}".format(row_overlaps[0], row_overlaps[-1], col_overlaps[0],
                                                          col_overlaps[-1])
            for row_nr in range(row_overlaps[0], row_overlaps[-1] + 1):
                for col_nr in range(col_overlaps[0], col_overlaps[-1] + 1):
                    existing_grid_cells.add((row_nr, col_nr))
        else:
            normal_cells_delete_ids.add(cell_ann['id'])
    if len(normal_cells_delete_ids) > 0:
        logger.debug('deleting {} detected normal cells (using row/col intersections instead'.format(
            len(normal_cells_delete_ids)))
    annotations = [ann for ann in annotations if ann['id'] not in normal_cells_delete_ids]

    # Create non-multi cells from row/column intersections, instead of using detection outputs
    new_cell_anns = []
    for col_nr in sorted(list(col_anns.keys())):
        for row_nr in sorted(list(row_anns.keys())):
            col_ann = col_anns[col_nr]
            row_ann = row_anns[row_nr]
            intsct_x0 = col_ann['bbox'][0]
            intsct_x1 = col_ann['bbox'][0] + col_ann['bbox'][2]
            intsct_y0 = row_ann['bbox'][1]
            intsct_y1 = row_ann['bbox'][1] + row_ann['bbox'][3]
            page = row_ann['page']
            bbox_from_intersection = [intsct_x0, intsct_y0, intsct_x1 - intsct_x0, intsct_y1 - intsct_y0]
            row_start = row_nr
            row_end = row_nr
            col_start = col_nr
            col_end = col_nr

            grid_coord = (row_nr, col_nr)
            if grid_coord not in existing_grid_cells:
                # new_cell_id = get_new_ann_id()
                new_cell_ann = {'category': 'table_cell'}
                new_cell_ann['row_range'] = [row_start, row_end]
                new_cell_ann['col_range'] = [col_start, col_end]
                new_cell_ann['properties'] = "{}-{},{}-{}".format(row_start, row_end, col_start, col_end)
                new_cell_ann['bbox'] = bbox_from_intersection

                existing_grid_cells.add(grid_coord)
                new_cell_anns.append(new_cell_ann)
    annotations += new_cell_anns

    # expand multi-cells such that they are aligned with their corresponding rows/columns
    for ann in annotations:
        if ann['category'] == 'table_cell':
            [row_start, row_end] = ann['row_range']
            [col_start, col_end] = ann['col_range']
            if row_start == row_end and col_start == col_end:
                continue
            row_bot = row_anns[row_start]
            row_top = row_anns[row_end]
            cell_bbox = ann['bbox']
            new_y0 = row_bot['bbox'][1]
            new_y1 = row_top['bbox'][1] + row_top['bbox'][3]

            ann['bbox'] = [cell_bbox[0], new_y0, cell_bbox[2], new_y1 - new_y0]
            col_left = col_anns[col_start]
            col_right = col_anns[col_end]
            cell_bbox = ann['bbox']
            new_x0 = col_left['bbox'][0]
            new_x1 = col_right['bbox'][0] + col_right['bbox'][2]
            ann['bbox'] = [new_x0, cell_bbox[1], new_x1 - new_x0, cell_bbox[3]]

    return annotations


def second_bbox_contained_in_first_bbox_xyformat(b1, b2, tolerance=0):
    return b1[0] - tolerance <= b2[0] and b1[1] - tolerance <= b2[1] and b1[2] + tolerance >= b2[2] and b1[
        3] + tolerance >= b2[3]


def second_bbox_contained_in_first_bbox(b1, b2, tolerance=0):
    return b1[0] - tolerance <= b2[0] and b1[1] - tolerance <= b2[1] and b1[0] + b1[2] + tolerance >= b2[0] + b2[2] and \
           b1[1] + b1[3] + tolerance >= b2[1] + b2[3]


def has_height_overlap_with_first_bbox_cocoformat(b1, b2, percentage=0.6):
    # tall reference annotation, e.g. row
    [struct_x0, struct_y0, struct_w, struct_h] = b1
    struct_y1 = struct_y0 + struct_h

    # small annotation
    [x0, y0, w, h] = b2
    y1 = y0 + h

    if y1 <= struct_y0:
        return False
    elif y0 >= struct_y1:
        return False
    elif struct_h == 0 or h == 0:
        return False
    overlap_y0 = max(y0, struct_y0)
    overlap_y1 = min(y1, struct_y1)
    overlap_height = overlap_y1 - overlap_y0
    if overlap_height / struct_h > percentage:
        return True
    elif overlap_height / h > percentage:
        # logger.debug('accept overlap for cell that overlaps more than {} of its own height'.format(percentage))
        return True


def has_width_overlap_with_first_bbox_cocoformat(b1, b2, percentage=0.6):
    [x0, y0, w, h] = b2
    x1 = x0 + w
    [struct_x0, struct_y0, struct_w, struct_h] = b1
    struct_x1 = struct_x0 + struct_w
    if x1 <= struct_x0:
        return False
    elif x0 >= struct_x1:
        return False
    elif struct_w == 0 or w == 0:
        return False
    overlap_x0 = max(x0, struct_x0)
    overlap_x1 = min(x1, struct_x1)
    overlap_width = overlap_x1 - overlap_x0
    if overlap_width / struct_w > percentage:
        return True
    elif overlap_width / w > percentage:
        # logger.debug('accept overlap for cell that overlaps more than {} of its own width'.format(percentage))
        return True


def adjust_bboxes_to_true_page_coordinates(current_region_annotations, crop_x0, crop_y0):
    # correct all bboxes with crop information
    for ann in current_region_annotations:
        ann_bbox = ann['bbox']
        ann_bbox_corrected = [ann_bbox[0] + crop_x0, ann_bbox[1] + crop_y0, ann_bbox[2], ann_bbox[3]]
        ann['bbox'] = ann_bbox_corrected


def assign_numbers_to_rows_and_cols(current_region_annotations):
    ann_by_id = dict()
    for ann in current_region_annotations:
        ann_by_id[ann['id']] = ann

    all_rows = [ann for ann in current_region_annotations if ann['category'] == 'table_row']
    all_cols = [ann for ann in current_region_annotations if ann['category'] == 'table_col']
    all_cells = [ann for ann in current_region_annotations if ann['category'] == 'table_cell']
    assert (len(all_rows) + len(all_cols) + len(all_cells)) == len(current_region_annotations)
    rows_sorted = sorted(all_rows, key=lambda x: x['bbox'][1] + (x['bbox'][3] / 2.0))
    cols_sorted = sorted(all_cols, key=lambda x: x['bbox'][0] + (x['bbox'][2] / 2.0))

    for row_nr, row in enumerate(rows_sorted):
        ann_by_id[row['id']]['row_nr'] = row_nr
    for col_nr, col in enumerate(cols_sorted):
        ann_by_id[col['id']]['col_nr'] = col_nr

    num_rows = len(rows_sorted)
    num_cols = len(cols_sorted)
    return num_rows, num_cols


def limit_all_annotations_to_crop_region(structured_annotations, crop_bbox):
    [crop_x0, crop_y0, crop_w, crop_h] = crop_bbox
    crop_x1 = crop_x0 + crop_w
    crop_y1 = crop_y0 + crop_h
    for ann in structured_annotations:
        [x0, y0, w, h] = ann['bbox']
        x1 = x0 + w
        y1 = y0 + h
        x0 = max(x0, crop_x0)
        x1 = min(x1, crop_x1)
        y0 = max(y0, crop_y0)
        y1 = min(y1, crop_y1)
        new_w = x1 - x0
        new_h = y1 - y0
        ann['bbox'] = [x0, y0, new_w, new_h]
    return structured_annotations


def resolve_direct_nesting_of_rows_and_columns(annotations):
    all_rows = [ann for ann in annotations if ann['category'] == 'table_row']
    all_cols = [ann for ann in annotations if ann['category'] == 'table_col']

    contain_id_mapping = defaultdict(list)
    if len(all_rows) > 1:
        for i, row in enumerate(all_rows):
            for other_row in all_rows[i + 1:]:
                if second_bbox_contained_in_first_bbox(row['bbox'], other_row['bbox']):
                    contain_id_mapping[row['id']].append(other_row['id'])
                if second_bbox_contained_in_first_bbox(other_row['bbox'], row['bbox']):
                    contain_id_mapping[other_row['id']].append(row['id'])
    if len(all_cols) > 1:
        for i, ann in enumerate(all_cols):
            for other_ann in all_cols[i + 1:]:
                if second_bbox_contained_in_first_bbox(ann['bbox'], other_ann['bbox']):
                    contain_id_mapping[ann['id']].append(other_ann['id'])
                if second_bbox_contained_in_first_bbox(other_ann['bbox'], ann['bbox']):
                    contain_id_mapping[other_ann['id']].append(ann['id'])

    remove_ids = set()
    for row_containing_others_id, list_of_contained in contain_id_mapping.items():
        if len(list_of_contained) > 1:
            logger.debug('removing row/column that contains {} other rows/columns'.format(len(list_of_contained)))
            remove_ids.add(row_containing_others_id)
        elif len(list_of_contained) == 1:
            remove_ids.add(list_of_contained[0])

    len_before = len(annotations)
    annotations = [ann for ann in annotations if ann['id'] not in remove_ids]
    len_after = len(annotations)
    logger.debug('removed {} directly nested row/column annotations'.format(len_before - len_after))
    return annotations


def process_all_table_structure_annotations(input_annotation_list):
    current_region_annotations = input_annotation_list
    current_region_annotations = resolve_direct_nesting_of_rows_and_columns(current_region_annotations)
    _, _ = assign_numbers_to_rows_and_cols(current_region_annotations)

    structured_annotations = copy.deepcopy(current_region_annotations)
    structured_annotations = adjust_and_move_row_and_column_borders(structured_annotations)

    structured_annotations = create_all_cells_and_assign_coordinates(structured_annotations)
    return structured_annotations
