import argparse
import json
import logging
import os

import skimage.io

from docparser import stage1_entity_detector
from docparser import stage2_structure_parser
from docparser.utils.data_utils import create_dir_if_not_exists, find_available_documents, \
    create_eval_output_dir, generate_path_to_eval_dir, DocsDataset, get_dirname_for_path
from docparser.utils.eval_utils import generate_obj_detection_results_based_on_directories, \
    generate_relation_classification_results_based_on_directories, update_with_mAP_singlemodel, \
    convert_bbox_list_to_save_format, convert_table_structure_list_to_save_format, evaluate_icdar_xmls

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)
import tqdm


def detect_icdar_structures(output_dir, detections_postfix, dataset_dir, crop_dataset_dir, dataset_ann_version,
                            elems_dir):
    detection_files_dir = os.path.join(output_dir, 'detections' + '_' + detections_postfix)
    structure_parser = stage2_structure_parser.StructureParser()
    output_xml_dir = os.path.join(output_dir, 'structure_xml_icdar')
    create_dir_if_not_exists(output_xml_dir)
    structure_parser.create_icdar_structures_for_docs(detection_files_dir, output_xml_dir, dataset_dir,
                                                      crop_dataset_dir, dataset_ann_version, elems_dir)
    return output_xml_dir


def evaluate_detections(output_dir, detections_postfix, gt_postfix, gt_dir, evaluate_structure=False):
    result_dir = os.path.join(output_dir, 'results_{}'.format(detections_postfix))
    create_dir_if_not_exists(result_dir)
    detections_results_json_path = os.path.join(result_dir,
                                                'detection_results.json')
    relations_results_json_path = os.path.join(result_dir,
                                               'relation_results.json')
    detection_files_dir = os.path.join(output_dir, 'detections' + '_' + detections_postfix)
    relations_files_dir = os.path.join(detection_files_dir, 'relations')

    if evaluate_structure is False:
        detections_gtFolder = os.path.join(gt_dir, 'groundtruths_{}'.format(gt_postfix))
        _, result_dict = generate_obj_detection_results_based_on_directories(detection_files_dir, detections_gtFolder,
                                                                             detections_results_json_path,
                                                                             return_result_dict=True)
        update_with_mAP_singlemodel(result_dict, None)
        for iou in ['0.5', '0.65', '0.8']:
            logger.info('mAP for iou={}: {}'.format(iou, result_dict['iou'][iou]['mAP']))
            other_APs = [(x['class'], x['AP']) for x in result_dict['iou'][iou]['detections']]
            # logger.info('other APs: {}'.format(other_APs))
    else:
        relations_gt_file = os.path.join(gt_dir, 'groundtruths_{}_relations.json'.format(gt_postfix))
        _, result_dict = generate_relation_classification_results_based_on_directories(relations_files_dir,
                                                                                       detections_results_json_path,
                                                                                       relations_results_json_path,
                                                                                       relations_gt_file,
                                                                                       return_result_dict=True)

        for iou in ['0.5', '0.65', '0.8']:
            logger.info('F1 for iou={}: {}'.format(iou, result_dict['iou'][iou]['f1']))


def create_entity_detections_for_inputs(output_dir, entity_detector, entry_tuples):
    create_dir_if_not_exists(output_dir)
    image_ids = []
    for entry_nr, entry_tuple in tqdm.tqdm(enumerate(entry_tuples), total=len(entry_tuples)):
        (example_id, dataset_dir, _) = entry_tuple

        image_ids.append(example_id)
        page = 0
        image_filepath = os.path.join(dataset_dir, example_id, example_id + '-' + str(page) + '.png')
        image_name = os.path.basename(image_filepath)
        image = skimage.io.imread(image_filepath)
        entity_predictions = entity_detector.predict(image)

        det_dir = create_eval_output_dir(output_dir)
        detections_textfile = os.path.join(image_name + '.txt')
        entity_detector.save_predictions_to_file(entity_predictions, det_dir, detections_textfile)


def detect_structures_for_inputs(output_dir, dataset_dir=None, detections_postfix='origimg', do_postprocessing=False,
                                 postprocessed_postfix='postprocessed', debug_gui_dir=None, debug_gui_tag='wsft160',
                                 table_mode=False):
    detection_files_dir = os.path.join(output_dir, 'detections' + '_' + detections_postfix)
    create_dir_if_not_exists(detection_files_dir)
    structure_parser = stage2_structure_parser.StructureParser()
    detection_files = [os.path.join(detection_files_dir, x) for x in os.listdir(detection_files_dir) if
                       x.endswith('.png.txt')]

    time_per_structure_detection = []
    for detection_file in detection_files:
        img_relations_dict = structure_parser.create_structure_for_doc(detection_file, table_mode=table_mode,
                                                                       do_postprocessing=do_postprocessing)
        if do_postprocessing:
            detection_files_dir_postprocessing = detection_files_dir.replace(detections_postfix,
                                                                             '') + postprocessed_postfix
            create_dir_if_not_exists(detection_files_dir_postprocessing)
            if table_mode is False:
                predictions_dict = convert_bbox_list_to_save_format(img_relations_dict['all_bboxes'])
            elif table_mode is True:
                predictions_dict = convert_table_structure_list_to_save_format(
                    img_relations_dict['table_structure_annotations'])
            detection_filename = os.path.basename(detection_file)

            stage1_entity_detector.EntityDetector.write_predictions_to_file(predictions_dict,
                                                                            detection_files_dir_postprocessing,
                                                                            detection_filename)
            relations_subdir = os.path.join(detection_files_dir_postprocessing, 'relations')
        else:
            relations_subdir = os.path.join(detection_files_dir, 'relations')

        # NOTE: we do not create relation triples for tables (structure is evaluated for ICDAR via xml files)
        if not table_mode:
            create_dir_if_not_exists(relations_subdir)
            relations_filename = os.path.basename(detection_file).replace('.png.txt', 'png.txt_relations.json')
            relations_path = os.path.join(relations_subdir, relations_filename)
            logger.debug('saving relations to {}'.format(relations_path))
            with open(relations_path, 'w') as out_file:
                json.dump(img_relations_dict['relations'], out_file, indent=1)

        if debug_gui_dir is not None:
            img_name = os.path.basename(detection_file.replace('.png.txt', '.png'))
            img_id = img_name.replace('-0.png', '')
            img_path = os.path.join(dataset_dir, img_id, img_name)
            flat_annotation_list = img_relations_dict['flat_annotations']
            structure_parser.create_gui_doc_entry(debug_gui_dir, flat_annotation_list, img_path, out_tag=debug_gui_tag)

    return time_per_structure_detection


def demo_table_detection(repo_root_dir):
    logger.info("Starting table structure detection demo..")
    logger.info(
        "default models correspond to best models for iou=0.5. accuracies for other ious might differ from reported numbers in paper")
    logger.info(
        "wsft model result might slightly differ from paper, as we average our results over 3 separate runs in the paper")
    logger.info("Note that here we evaluate mAP on table rows, columns, as well as multi-row/column + header cells")
    eval_root = os.path.join(repo_root_dir, 'data/eval')
    model_log_dir = os.path.join(repo_root_dir, 'data/logs')
    dataset_dirs = {
        'dev': os.path.join(repo_root_dir, 'datasets/arxivdocs_target/splits/by_tabular/dev'),
        'test': os.path.join(repo_root_dir, 'datasets/arxivdocs_target/splits/by_tabular/test')
    }

    gt_dirs = {'test': os.path.join(repo_root_dir, 'datasets/arxivdocs_target/splits/by_tabular/flat_lists/test'),
               'dev': os.path.join(repo_root_dir, 'datasets/arxivdocs_target/splits/by_tabular/flat_lists/dev')
               }
    splits = ['dev', 'test']

    models_to_eval = ['lowlevel_wsft', 'lowlevel_ws', 'lowlevel_manual']
    annotation_tag = 'cleanGT'
    detections_postfix = 'origimg'
    postprocessed_postfix = 'origimg_postpr'

    for split in splits:
        dataset_dir = dataset_dirs[split]
        gt_dir = gt_dirs[split]

        for model_to_eval in models_to_eval:
            output_dir = generate_path_to_eval_dir(eval_root, dataset_dir, model_to_eval, root_dir=repo_root_dir)
            entry_tuples = find_available_documents(dataset_dir=dataset_dir, version=annotation_tag)

            entity_detector = stage1_entity_detector.EntityDetector(detection_max_instances=200)
            entity_detector.init_model(model_log_dir=model_log_dir, default_weights=model_to_eval)
            logger.info('Generate entity predictions for {}...'.format(model_to_eval))
            create_entity_detections_for_inputs(output_dir, entity_detector, entry_tuples)

            detect_structures_for_inputs(output_dir, table_mode=True, dataset_dir=dataset_dir, do_postprocessing=True,
                                         postprocessed_postfix=postprocessed_postfix)
            logger.info('---------------')
            logger.info('Table entity detection')
            logger.info('detections for {} on {} set:'.format(model_to_eval, split))
            evaluate_detections(output_dir, detections_postfix=detections_postfix, gt_postfix=detections_postfix,
                                gt_dir=gt_dir, evaluate_structure=False)
            logger.info('---------------\n')


def demo_icdar(repo_root_dir):
    logger.info("Starting ICDAR 2013 table structure demo..")
    eval_root = os.path.join(repo_root_dir, 'data/eval')
    model_log_dir = os.path.join(repo_root_dir, 'data/logs')
    dataset_dirs = {
        'dev': os.path.join(repo_root_dir, 'datasets/icdar2013_docparser/mixed_icdar_crop_splits/icdar_dev/'),
        'test': os.path.join(repo_root_dir, 'datasets/icdar2013_docparser/mixed_icdar_crop_splits/icdar_test/')
    }
    gt_dirs = {
        'dev': os.path.join(repo_root_dir,
                            'datasets/icdar2013_docparser/mixed_icdar_crop_splits/flat_lists/icdar_dev/'),
        'test': os.path.join(repo_root_dir,
                             'datasets/icdar2013_docparser/mixed_icdar_crop_splits/flat_lists/icdar_test/')
    }
    uncropped_dataset_dirs = {
        'dev': os.path.join(repo_root_dir, 'datasets/icdar2013_docparser/mixed_icdar_splits/icdar_dev/'),
        'test': os.path.join(repo_root_dir, 'datasets/icdar2013_docparser/mixed_icdar_splits/icdar_test/')
    }
    xml_gt_dirs = {
        'dev': os.path.join(repo_root_dir,
                            'datasets/icdar2013_docparser/extra_files/structure/processed_files/subset_dev/GT_corrected'),
        'test': os.path.join(repo_root_dir,
                             'datasets/icdar2013_docparser/extra_files/structure/processed_files/subset_test/GT_corrected')
    }

    # xml files with words+positions provided by nurminen
    elems_folder = os.path.join(repo_root_dir,
                                'datasets/icdar2013_docparser/extra_files/structure/processed_files/mixed_icdar/elems/')

    splits = ['dev', 'test']
    models_to_eval = ['icdar_wsft']
    annotation_tag = 'cleanGT'
    detections_postfix = 'origimg'

    nurminen_xml_outputs = os.path.join(repo_root_dir,
                                        'datasets/icdar2013_docparser/extra_files/structure/processed_files/mixed_icdar/nurminen_comp-output_corrected')
    jar_file_path = os.path.join(repo_root_dir, 'docparser/utils/dataset-tools-20180206.jar')

    for split in splits:
        dataset_dir = dataset_dirs[split]
        uncropped_dataset_dir = uncropped_dataset_dirs[split]
        gt_dir = gt_dirs[split]
        xml_gt_dir = xml_gt_dirs[split]

        # logger.info('evaluating {}'.format(dataset_dir))
        for model_to_eval in models_to_eval:
            output_dir = generate_path_to_eval_dir(eval_root, dataset_dir, model_to_eval, root_dir=repo_root_dir)
            logger.info('output dir: {}'.format(output_dir))
            entry_tuples = find_available_documents(dataset_dir=dataset_dir, version=annotation_tag)
            entity_detector = stage1_entity_detector.EntityDetector(resize_mode='square',
                                                                    detection_max_instances=200)
            entity_detector.init_model(model_log_dir=eval_root, default_weights=model_to_eval)
            create_entity_detections_for_inputs(output_dir, entity_detector, entry_tuples)

            xml_output_dir = detect_icdar_structures(output_dir, detections_postfix, uncropped_dataset_dir, dataset_dir,
                                                     annotation_tag, elems_folder)

            results_dict = evaluate_icdar_xmls(xml_output_dir, xml_gt_dir, jar_file_path)
            logger.info('---------------')
            logger.info('icdar f1 score for {} on {}: {}'.format(model_to_eval, split, results_dict['f1']))
            logger.info('---------------\n')

    test_xml_gt_dir = xml_gt_dirs['test']
    results_dict = evaluate_icdar_xmls(nurminen_xml_outputs, test_xml_gt_dir, jar_file_path)
    logger.info('---------------')
    logger.info('icdar f1 score, nurminen baseline on test ({}: {}'.format(test_xml_gt_dir, results_dict['f1']))
    logger.info('---------------\n')


def demo_page_detection(repo_root_dir):
    logger.info("Starting full page parsing demo..")
    logger.info(
        "default models correspond to best models for iou=0.5. accuracies for other ious might differ from reported numbers in paper")
    logger.info(
        "wsft model result might slightly differ from paper, as we average our results over 3 separate runs in the paper")
    eval_root = os.path.join(repo_root_dir, 'data/eval')
    model_log_dir = os.path.join(repo_root_dir, 'data/logs')

    splits = ['dev', 'test']
    dataset_dirs = {
        'dev': os.path.join(repo_root_dir, 'datasets/arxivdocs_target/splits/by_page/dev'),
        'test': os.path.join(repo_root_dir, 'datasets/arxivdocs_target/splits/by_page/test')
    }

    gt_dirs = {
        'dev': os.path.join(repo_root_dir, 'datasets/arxivdocs_target/splits/by_page/flat_lists/dev'),
        'test': os.path.join(repo_root_dir, 'datasets/arxivdocs_target/splits/by_page/flat_lists/test')
    }

    models_to_eval = ['highlevel_wsft', 'highlevel_ws', 'highlevel_manual']
    annotation_tag = 'cleanGT'
    detections_postfix = 'origimg'
    postprocessed_postfix = 'origimg_postpr'

    for split in splits:
        dataset_dir = dataset_dirs[split]
        gt_dir = gt_dirs[split]
        for model_to_eval in models_to_eval:
            output_dir = generate_path_to_eval_dir(eval_root, dataset_dir, model_to_eval, root_dir=repo_root_dir)
            entry_tuples = find_available_documents(dataset_dir=dataset_dir, version=annotation_tag)
            entity_detector = stage1_entity_detector.EntityDetector()
            entity_detector.init_model(model_log_dir=model_log_dir, default_weights=model_to_eval)

            create_entity_detections_for_inputs(output_dir, entity_detector, entry_tuples)
            detect_structures_for_inputs(output_dir, dataset_dir=dataset_dir, do_postprocessing=False)
            detect_structures_for_inputs(output_dir, dataset_dir=dataset_dir, do_postprocessing=True,
                                         postprocessed_postfix=postprocessed_postfix)

            logger.info('---------------')
            logger.info('Full-page parsing')
            logger.info('detections for {} on {} set:'.format(model_to_eval, split))
            evaluate_detections(output_dir, detections_postfix=detections_postfix, gt_postfix=detections_postfix,
                                gt_dir=gt_dir, evaluate_structure=False)
            logger.info('detections refined:')
            evaluate_detections(output_dir, detections_postfix=postprocessed_postfix, gt_postfix=detections_postfix,
                                gt_dir=gt_dir, evaluate_structure=False)
            logger.info('structure:')
            evaluate_detections(output_dir, detections_postfix=detections_postfix, gt_postfix=detections_postfix,
                                gt_dir=gt_dir, evaluate_structure=True)
            logger.info('structure refined:')
            evaluate_detections(output_dir, detections_postfix=postprocessed_postfix, gt_postfix=detections_postfix,
                                gt_dir=gt_dir, evaluate_structure=True)
            logger.info('---------------\n')


def demo_finetune(repo_root_dir):
    logger.info("Starting example training run for page entity detection, initialized with DocParser WS..")
    logger.info("Models are saved to data/eval/logs")
    # Example method to perform training/finetuning of our DocParser
    model_log_dir = os.path.join(repo_root_dir, 'data/logs')
    train_dataset_dir = os.path.join(repo_root_dir, 'datasets/arxivdocs_target/splits/by_page/train')
    val_dataset_dir = os.path.join(repo_root_dir, 'datasets/arxivdocs_target/splits/by_page/dev')
    annotation_tag = 'cleanGT'

    # prepare datasets
    dataset_train = DocsDataset()
    highlevel_classes = dataset_train.DEFAULT_HIGHLEVEL_CLASSES
    dataset_train.load_docs(train_dataset_dir, annotation_tag, classes=highlevel_classes)
    dataset_train.prepare()
    dataset_val = DocsDataset()
    dataset_val.load_docs(val_dataset_dir, annotation_tag, classes=highlevel_classes)
    dataset_val.prepare()

    entity_detector = stage1_entity_detector.EntityDetector()
    # Initialize detector with weights from weakly supervised model
    entity_detector.init_model(model_log_dir=model_log_dir, default_weights='coco', train_mode=True)
    entity_detector.train(dataset_train, dataset_val)


#    #trained/finetuned model could later be used for detection:
#    entity_detector = stage1_entity_detector.EntityDetector()
#    weights_path = os.path.join(model_log_dir, 'docparser_defaultTIMESTAMP', 'mask_rcnn_docparser_default_XXXX.h5')
#    entity_detector.init_model(model_log_dir=model_log_dir, custom_weights=weights_path, train_mode=False)
#    entry_tuples = find_available_documents(dataset_dir=os.path.join(repo_root_dir, 'datasets/arxivdocs_target/splits/by_page/dev'), version='cleanGT')
#    output_dir= os.path.join(repo_root_dir, 'data/tmp')
#    create_entity_detections_for_inputs(output_dir, entity_detector, entry_tuples)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='DocParser Demoscript')
    parser.add_argument('--page', action='store_true',
                        help='Run and evaluate DocParser default models for page structure parsing on arXivDocs-target')
    parser.add_argument('--table', action='store_true',
                        help='Run and evaluate DocParser default models for table entity detection on arXivDocs-target')
    parser.add_argument('--icdar', action='store_true',
                        help='Run and evaluate DocParser finetuned model for table structure parsing on ICDAR 2013')
    parser.add_argument('--finetune', action='store_true',
                        help='Demo fine-tuning of DocParser ws on arXivDocs-target')
    args = parser.parse_args()

    cwd_path = os.getcwd()
    demo_dir = 'DocParser'
    current_dir = get_dirname_for_path(cwd_path)
    try:
        assert demo_dir == current_dir
    except AssertionError as e:
        logger.error(
            "Please run script from the 'docparser' directory (/PATH_TO_CODE/emnlp_codes/docparser), current dir: {}".format(
                current_dir))
        raise

    if not any([args.page, args.table, args.icdar, args.finetune]):
        print(parser.print_help())

    if args.page:
        demo_page_detection(cwd_path)
    if args.table:
        demo_table_detection(cwd_path)
    if args.icdar:
        demo_icdar(cwd_path)

    # finetuning code snippet for demonstration - not further used in this demo for evaluation
    if args.finetune:
        demo_finetune(cwd_path)
