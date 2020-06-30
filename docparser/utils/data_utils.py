import json
import logging
import os
from collections import defaultdict

import mrcnn.utils as mrcnn_utils
import numpy as np
import random
import skimage
from PIL import Image

logger = logging.getLogger(__name__)


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, set):
            return list(obj)
        else:
            return super(CustomJSONEncoder, self).default(obj)


def get_all_children_ids_with_child_dictionary(parent_id, ann_children_ids):
    final_new_children_ids = []
    for new_child_id in ann_children_ids[parent_id]:
        final_new_children_ids.append(new_child_id)
        final_new_children_ids += get_all_children_ids_with_child_dictionary(new_child_id, ann_children_ids)
    return final_new_children_ids


def represents_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def create_dir_if_not_exists(dir_path):
    if not os.path.isdir(dir_path):
        logger.debug('creating directory: {}'.format(dir_path))
        try:
            os.makedirs(dir_path)
        except FileExistsError:
            logger.error('paralleliziation error')


def _draw_rectangle_polygon(r0, c0, width, height):
    rr, cc = [r0, r0 + width, r0 + width, r0], [c0, c0, c0 + height, c0 + height]
    return skimage.draw.polygon(rr, cc)


def get_all_children_ids_with_child_dictionary(parent_id, ann_children_ids):
    final_new_children_ids = []
    for new_child_id in ann_children_ids[parent_id]:
        final_new_children_ids.append(new_child_id)
        final_new_children_ids += get_all_children_ids_with_child_dictionary(new_child_id, ann_children_ids)
    return final_new_children_ids


def get_all_bbox_anns_for_current_id(parent_ann_id, ann_children_ids, ann_by_id, page=None):
    all_children_ids = get_all_children_ids_with_child_dictionary(parent_ann_id, ann_children_ids)
    if page is None:
        all_bbox_anns = [ann_by_id[ann_id] for ann_id in all_children_ids if ann_by_id[ann_id]['category'] == 'box']
    else:
        all_bbox_anns = [ann_by_id[ann_id] for ann_id in all_children_ids if
                         ann_by_id[ann_id]['category'] == 'box' and ann_by_id[ann_id]['page'] == page]

    return all_bbox_anns


def get_merged_bbox_from_list_of_bbox_anns(bbox_anns):
    xmin, ymin, xmax, ymax = 10000, 10000, -1, -1
    for ann in bbox_anns:
        try:
            [x0, y0, w, h] = ann['bbox']
        except KeyError as e:
            logger.error('KeyError when trying to get bbox from ann: {}'.format(ann))
            raise
        x1 = x0 + w
        y1 = y0 + h
        if x0 < xmin:
            xmin = x0
        if y0 < ymin:
            ymin = y0
        if x1 > xmax:
            xmax = x1
        if y1 > ymax:
            ymax = y1
    result_w = xmax - xmin
    result_h = ymax - ymin
    return [xmin, ymin, result_w, result_h]


def create_annotations_to_add(annotations_path, only_multicells, only_labelcells, classes, page):
    try:
        annotations_list = json.load(open(annotations_path))
    except FileNotFoundError:
        logger.error('no annotation file found at {}'.format(annotations_path))
        raise

    if only_multicells is True or only_labelcells is True:
        if 'table_cell' not in classes:
            raise NotImplementedError('table_cell class must be active when selecting only multicells')
        multicell_count = 0
        new_annotation_list = []
        for ann in annotations_list:
            if ann['category'] == 'table_cell':
                col_range = ann['col_range']
                row_range = ann['row_range']
                if only_multicells is True and col_range is not None and col_range[0] is not None and (
                        col_range[1] > col_range[0]):
                    multicell_count += 1
                    new_annotation_list.append(ann)
                elif only_multicells is True and row_range is not None and row_range[0] is not None and (
                        row_range[1] > row_range[0]):
                    multicell_count += 1
                    new_annotation_list.append(ann)
                elif only_labelcells is True and (
                        (col_range is not None and col_range[0] is not None and col_range[0] == 0) or (
                        row_range is not None and row_range[0] is not None and row_range[0] == 0)):
                    new_annotation_list.append(ann)
            else:
                new_annotation_list.append(ann)
        annotations_list = new_annotation_list

    annotations = annotations_list
    ann_by_id = dict()
    anns_by_cat = defaultdict(list)
    ann_children_ids = defaultdict(list)

    for ann in annotations:
        ann_by_id[ann['id']] = ann
        anns_by_cat[ann['category']].append(ann)
        ann_children_ids[ann['parent']].append(ann['id'])

    returned_list_of_anns = []
    for annotation_class in classes:
        for annotation_of_allowed_class in anns_by_cat[annotation_class]:

            all_bboxes_for_current_annotation = get_all_bbox_anns_for_current_id(annotation_of_allowed_class['id'],
                                                                                 ann_children_ids, ann_by_id, page)
            if len(all_bboxes_for_current_annotation) == 0:
                logger.warning('no bbox found for {} in {}'.format(annotations_path))
                continue

            merged_bbox = get_merged_bbox_from_list_of_bbox_anns(all_bboxes_for_current_annotation)
            if merged_bbox[2] <= 0 or merged_bbox[3] <= 0:
                logger.warning('bbox for image is zero height or width, skipping..')
                continue
            annotation_of_allowed_class['bbox'] = merged_bbox
            returned_list_of_anns.append(annotation_of_allowed_class)

    return returned_list_of_anns


def get_random_entry_subset(entry_tuples, manualseed, subset_numimgs):
    logger.debug('using manual random seed: {}'.format(manualseed))
    random.seed(manualseed)
    entry_random_subset = random.sample(entry_tuples, subset_numimgs)
    logger.debug('subselected {} random images from total of {}:'.format(subset_numimgs, len(entry_random_subset),
                                                                         entry_random_subset))
    return entry_random_subset


def find_available_documents(dataset_dir, version, subset_sample=False, subset_numimgs=100000, manualseed=None):
    # Load annotations
    entry_tuples = []
    for entry in os.scandir(dataset_dir):
        if not entry.name.startswith('.') and entry.is_dir():
            entry_tuples.append((entry.name, dataset_dir, version))
    entry_tuples = sorted(entry_tuples)

    if subset_sample:
        if manualseed is not None and subset_numimgs is not None:
            entry_tuples_rd_subset = get_random_entry_subset(entry_tuples, manualseed, subset_numimgs)
            entry_tuples = entry_tuples_rd_subset
        else:
            raise NotImplementedError

    entry_tuples = sorted(entry_tuples)

    return entry_tuples


def gather_and_sanity_check_anns(generated_annotations):
    annotations = []

    all_unique_ids = set()
    ann_by_id = dict()
    for ann in generated_annotations:
        if ann['id'] not in all_unique_ids:
            all_unique_ids.add(ann['id'])
            annotations.append(ann)
            ann_by_id[ann['id']] = ann
        elif ann_by_id[ann['id']] == ann:
            logger.warning(
                "WARNING: annotation ID for {} is a duplicate, annotation is skipped because previous occurrence is identical!".format(
                    ann))
        else:
            raise AttributeError("ERROR: annotation ID for {} is a duplicate!".format(ann))

    return annotations, ann_by_id


class DocsDataset(mrcnn_utils.Dataset):
    ALL_CLASSES = ['table', 'table_caption', 'content_line', 'tabular', 'table_cell', 'table_row', 'table_col',
                   'content_block', 'figure', 'figure_graphic', 'figure_caption', 'equation', 'equation_formula',
                   'equation_label', 'itemize', 'item', 'heading', 'abstract', 'bibliography', 'affiliation', 'title',
                   'author', 'bib_block', 'head', 'foot', 'date', 'subject', 'page_nr']
    DEFAULT_HIGHLEVEL_CLASSES = ['content_block', 'table', 'tabular', 'figure', 'heading', 'abstract', 'equation',
                                 'itemize', 'item', 'bib_block', 'table_caption', 'figure_graphic', 'figure_caption',
                                 'head', 'foot', 'page_nr', 'date', 'subject', 'author', 'affiliation']

    ALL_TABLE_SUBCLASSES = ['tabular', 'table_caption', 'table_row', 'table_col', 'table_cell']

    def load_docs(self, dataset_dir, version, classes=['table', 'table_caption', 'tabular', 'table_cell'],
                  subset_sample=False, subset_numimgs=100000, manualseed=None,
                  only_multicells=False, only_labelcells=False):
        """Load a subset of the Document dataset.
        dataset_dir: Root directory of the dataset.
        """
        added_img_counter = 0
        # Add classes
        for i, c in enumerate(self.ALL_CLASSES):
            self.add_class('docparser', i, c)

        entry_tuples = find_available_documents(dataset_dir=dataset_dir, version=version,
                                                subset_sample=subset_sample, subset_numimgs=subset_numimgs,
                                                manualseed=manualseed)

        for entry_nr, entry_tuple in enumerate(entry_tuples):
            (example_id, dataset_dir, version) = entry_tuple

            if entry_nr % 1000 == 0:
                logger.debug('loading entry {} of {}..'.format(entry_nr, len(entry_tuples)))

            annotations_path = os.path.join(dataset_dir, example_id, example_id + '-' + version + '.json')
            page = 0  # consider all images to be of page 0
            image_path = os.path.join(dataset_dir, example_id, example_id + '-' + str(page) + '.png')

            if len(classes) != len(set(classes)):
                logger.warning("WARNING: duplicates in input classes, removing duplicates..")
            classes = list(set(classes))
            self.add_image(
                "docparser",
                image_id=example_id + '-' + str(page),
                path=image_path,
                document=example_id, page=page,
                annotations_path=annotations_path,
                only_multicells=only_multicells,
                only_labelcells=only_labelcells,
                classes=classes)
            added_img_counter += 1
        logger.debug('added {} images to dataset'.format(added_img_counter))

    def image_reference(self, image_id):
        """Return the docs data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "docparser":
            return info["source"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a docs dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "docparser":
            return super(self.__class__, self).load_mask(image_id)

        image_path = image_info["path"]
        annotations_path = image_info["annotations_path"]
        only_multicells = image_info["only_multicells"]
        only_labelcells = image_info["only_labelcells"]
        classes = image_info["classes"]
        page = image_info["page"]

        try:
            im = Image.open(image_path)
        except OSError as e:
            logger.error("Error reading image: {}. skipping..".format(image_path))
            raise

        generated_annotations = create_annotations_to_add(annotations_path, only_multicells, only_labelcells, classes,
                                                          page)
        annotations, ann_by_id = gather_and_sanity_check_anns(generated_annotations)

        # create masks for image from annotations
        width, height = im.size
        mask = np.zeros([height, width, len(annotations)], dtype=np.uint8)
        bboxes = []
        ann_ids = []
        for i, ann in enumerate(annotations):
            bbox = ann['bbox']
            rr, cc = _draw_rectangle_polygon(bbox[1], bbox[0], bbox[3], bbox[2])
            try:
                mask[rr, cc, i] = 1
                new_bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                bboxes.append(new_bbox)
                ann_ids.append(ann['id'])
            except IndexError:
                logger.error(
                    'ERROR: while enerating mask for {}, because annotation bbox mismatch'.format(image_info['id']))
                raise

        class_ids = np.array([self.class_names.index(ann['category']) for ann in annotations])

        if len(annotations) == 0:
            logger.warning("WARNING: No annotations found")

        return mask.astype(np.bool), class_ids.astype(np.int32), bboxes, np.array(ann_ids).astype(np.int32)


def get_dirname_for_path(root_dir):
    repo_folder = os.path.basename(root_dir)
    if repo_folder == '':  # If root dir is given with backslash ending
        repo_folder = os.path.basename(os.path.dirname(root_dir))
    return repo_folder


def generate_path_to_eval_dir(eval_root_dir, dataset_dir, model_subdir, root_dir=None):
    if root_dir is None:
        docparser_root_position = dataset_dir.find('docparser/')
    else:
        repo_folder = get_dirname_for_path(root_dir)
        docparser_root_position = dataset_dir.find(repo_folder)
    if docparser_root_position == -1:
        raise NotImplementedError("Eval script is based on docparser evaluation")
    dataset_rel_path = dataset_dir[docparser_root_position:]
    dataset_eval_folder_name = dataset_rel_path.replace('/', '_')
    if dataset_eval_folder_name.endswith('_'):
        dataset_eval_folder_name = dataset_eval_folder_name[:-1]
    dataset_eval_folder_path = os.path.join(eval_root_dir, dataset_eval_folder_name)
    eval_dir = os.path.join(dataset_eval_folder_path, model_subdir)
    return eval_dir


def generate_path_to_epoch(eval_root_dir, dataset_dir, model_subdir, epoch_nr):
    eval_dir = generate_path_to_eval_dir(eval_root_dir, dataset_dir, model_subdir)
    eval_epoch_dir = os.path.join(eval_dir, str(epoch_nr))

    return eval_epoch_dir


def create_eval_output_dir(eval_epoch_dir, use_original_img_coords=True):
    if use_original_img_coords is False:
        raise NotImplementedError
    else:
        postfix = 'origimg'
        detections_origimg_dir = os.path.join(eval_epoch_dir, 'detections_{}'.format(postfix))
        create_dir_if_not_exists(detections_origimg_dir)
        return detections_origimg_dir
