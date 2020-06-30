import json
import logging
import os
import re
import subprocess
from collections import defaultdict

import statistics

from docparser.objdetmetrics_lib.BoundingBoxes import getBoundingBoxes
from docparser.objdetmetrics_lib.Evaluator import Evaluator, MethodAveragePrecision
from docparser.objdetmetrics_lib.utils import BBFormat
from docparser.utils.data_utils import CustomJSONEncoder

logger = logging.getLogger(__name__)


def generate_obj_detection_results_based_on_directories(detFolder, gtFolder, detections_epoch_json_result_path,
                                                        return_result_dict=False):
    epoch_results = {'iou': dict()}
    logger.debug("gathering bounding boxes for : {}".format(detFolder))
    allBoundingBoxes, allClasses = getBoundingBoxes(
        gtFolder, BBFormat.XYX2Y2, 'abs', isGT=True, header=True)
    allBoundingBoxes, allClasses = getBoundingBoxes(
        detFolder, BBFormat.XYX2Y2, 'abs', isGT=False, allBoundingBoxes=allBoundingBoxes, allClasses=allClasses,
        header=True)
    allClasses.sort()

    evaluator = Evaluator()
    ious = [0.5, 0.65, 0.8]
    for iou in ious:
        detections, matches_per_img = evaluator.GetPascalVOCMetrics(allBoundingBoxes, iou,
                                                                    method=MethodAveragePrecision.EveryPointInterpolation)
        epoch_results['iou'][str(iou)] = {'detections': detections, 'matches_per_img': matches_per_img}

    logger.debug("saving detection results to {}".format(detections_epoch_json_result_path))
    with open(detections_epoch_json_result_path, 'w') as out_file:
        json.dump(epoch_results, out_file, sort_keys=True, indent=1, cls=CustomJSONEncoder)

    success = True
    if return_result_dict:
        return success, epoch_results
    return success


def generate_relation_classification_results_based_on_directories(relation_files_dir, detections_epoch_json_result_path,
                                                                  relations_epoch_json_result_path, relations_gt_file,
                                                                  return_result_dict=False):
    with open(detections_epoch_json_result_path, 'r') as in_file:
        epoch_detection_results = json.load(in_file)

    with open(relations_gt_file, 'r') as in_file:
        relations_groundtruth_per_image = json.load(in_file)

    ious = ['0.5', '0.65', '0.8']
    all_detected_relation_files = [x for x in os.listdir(relation_files_dir) if x.endswith('.json')]
    detected_relations_per_image = dict()
    for relation_file_json in all_detected_relation_files:
        try:
            with open(os.path.join(relation_files_dir, relation_file_json), 'r') as in_file:
                detected_relations = json.load(in_file)
        except UnicodeDecodeError as e:
            logger.error("Cannot read file: {}".format(os.path.join(relation_files_dir, relation_file_json)))
            raise
        img_name = relation_file_json.replace('png.txt_relations.json', '.png')
        detected_relations_per_image[img_name] = detected_relations

    relations_results = get_matches_per_image(ious, detected_relations_per_image, epoch_detection_results,
                                              relations_groundtruth_per_image)

    with open(relations_epoch_json_result_path, 'w') as out_file:
        json.dump(relations_results, out_file, sort_keys=True, indent=1, cls=CustomJSONEncoder)

    success = True
    if return_result_dict:
        return success, relations_results
    return success


def get_matches_per_image(ious, detected_relations_per_image, epoch_detection_results, relations_groundtruth_per_image):
    relations_results = {'iou': dict()}

    relations_gathered = dict()
    for iou in ious:
        detected_object_matches_for_iou = epoch_detection_results['iou'][str(iou)]['matches_per_img']
        intersection = set(detected_relations_per_image.keys()).intersection(
            set(detected_object_matches_for_iou.keys()))
        images_without_any_detections = set(detected_relations_per_image.keys()) - intersection
        for image_without_any_detections in images_without_any_detections:
            # add emtpy dicts for images where no object was correctly detected
            detected_object_matches_for_iou[image_without_any_detections] = dict()

        assert set(detected_relations_per_image.keys()) == set(detected_object_matches_for_iou.keys())
        relations_results_iou = dict()
        detected_relations = []
        groundtruth_relations = []
        detected_to_gt_mappings = dict()
        for img_name in detected_relations_per_image.keys():
            detected_to_gt_mappings[img_name] = dict()
            for k, v in detected_object_matches_for_iou[img_name].items():
                detected_to_gt_mappings[img_name][str(v)] = str(k)
            for (rel_obj_id, rel_subj_id, rel_class) in detected_relations_per_image[img_name]:
                detected_relations.append((img_name, str(rel_obj_id), str(rel_subj_id), rel_class))
            for relation_tuple in relations_groundtruth_per_image[img_name]['relations']:
                (rel_obj_id, rel_subj_id, rel_class) = relation_tuple
                groundtruth_relations.append((img_name, str(rel_obj_id), str(rel_subj_id), rel_class))

        relations_gathered[iou] = {'detected_relations': set(detected_relations),
                                   'groundtruth_relations': set(groundtruth_relations),
                                   'id_mappings_for_detected_objects': detected_to_gt_mappings}

    # compare detected relations with GT relations
    for iou in ious:
        true_positives, false_positives, false_negatives = evaluate_relations_in_dict(relations_gathered[iou])
        precision, recall, f1 = calculate_relation_scores(true_positives, false_positives, false_negatives)
        _, _, f1_followed_by = calculate_relation_scores(true_positives, false_positives, false_negatives,
                                                         relation_type='COMES_BEFORE')
        _, _, f1_parent_of = calculate_relation_scores(true_positives, false_positives, false_negatives,
                                                       relation_type='IS_PARENT_OF')
        per_image_results = dict()
        for img_name in detected_relations_per_image.keys():
            img_true_positives = [x for x in true_positives if img_name == x[0]]
            img_false_positives = [x for x in false_positives if img_name == x[0]]
            img_false_negatives = [x for x in false_negatives if img_name == x[0]]
            _, _, img_f1 = calculate_relation_scores(img_true_positives, img_false_positives, img_false_negatives)
            per_image_results[img_name] = {'f1': img_f1}

        relations_results['iou'][iou] = {'true_positives': true_positives, 'false_positives': false_positives,
                                         'false_negatives': false_negatives, 'precision': precision, 'recall': recall,
                                         'f1': f1, 'f1_followed_by': f1_followed_by, 'f1_parent_of': f1_parent_of,
                                         'per_img_results': per_image_results}

    return relations_results


def calculate_relation_scores(true_positives, false_positives, false_negatives, relation_type=None):
    if relation_type is not None:
        true_positives = [x for x in true_positives if x[3] == relation_type]
        false_positives = [x for x in false_positives if x[3] == relation_type]
        false_negatives = [x for x in false_negatives if x[3] == relation_type]
    num_tp = len(true_positives)
    num_fp = len(false_positives)
    num_fn = len(false_negatives)
    if num_tp + num_fp == 0:
        precision = None
    else:
        precision = num_tp / (num_tp + num_fp)
    if num_tp + num_fn == 0:
        recall = None
    else:
        recall = num_tp / (num_tp + num_fn)
    if recall is None or precision is None or precision + recall == 0:
        logger.debug('precision/recall are zero!')
        f1 = 0
    else:
        f1 = 2 * ((precision * recall) / (precision + recall))
    return precision, recall, f1


def evaluate_relations_in_dict(relations_gathered):
    detected_relations = relations_gathered['detected_relations']
    groundtruth_relations = relations_gathered['groundtruth_relations']
    id_mappings_for_detected_objects = relations_gathered['id_mappings_for_detected_objects']

    detected_relations_mapped = set()
    for (img_name, rel_obj_id, rel_subj_id, rel_class) in detected_relations:
        # mark as not-matched (negative value)
        rel_obj_id_mapped = id_mappings_for_detected_objects[img_name].get(rel_obj_id,
                                                                           'NOMATCH_' + rel_obj_id)
        rel_subj_id_mapped = id_mappings_for_detected_objects[img_name].get(rel_subj_id,
                                                                            'NOMATCH_' + rel_subj_id)
        mapped_relation = (img_name, rel_obj_id_mapped, rel_subj_id_mapped, rel_class)
        detected_relations_mapped.add(mapped_relation)

    true_positives = detected_relations_mapped.intersection(groundtruth_relations)
    false_positives = detected_relations_mapped - groundtruth_relations
    false_negatives = groundtruth_relations - detected_relations_mapped
    assert true_positives.union(false_positives.union(false_negatives)) == detected_relations_mapped.union(
        groundtruth_relations)
    assert len(true_positives.intersection(false_positives)) == 0
    assert len(true_positives.intersection(false_negatives)) == 0
    assert len(false_positives.intersection(false_negatives)) == 0

    return true_positives, false_positives, false_negatives


def update_with_mAP_singlemodel(result_dict_for_ious, classes_to_consider):
    for iou, results_dict in result_dict_for_ious['iou'].items():
        all_APs = []
        all_classes = []
        for class_detections_dict in results_dict['detections']:
            current_class = class_detections_dict['class']
            # if classes_to_consider is None: get mAP over all classes
            if classes_to_consider is not None and current_class not in classes_to_consider:
                continue
            current_AP = float(class_detections_dict['AP'])
            all_APs.append(current_AP)
            all_classes.append(current_class)
        mAP = statistics.mean(all_APs)
        results_dict['mAP'] = mAP
        results_dict['mAP_classes'] = all_classes


def update_with_mAP(eval_results, classes_to_consider):
    for epoch_nr, iou_dict in eval_results['epochs'].items():
        update_with_mAP_singlemodel(iou_dict, classes_to_consider)


def convert_bbox_list_to_save_format(all_bboxes_for_img):
    prediction_dict = dict()

    if len(all_bboxes_for_img) == 0:
        prediction_dict['orig_img_shape'] = [None, None]
    else:
        (w, h) = all_bboxes_for_img[0].getImageSize()
        prediction_dict['orig_img_shape'] = [h, w]
    prediction_list = []
    for b in all_bboxes_for_img:
        # TODO: safety check whether 'bbox_orig_coords' or 'pred_bbox' should be used
        [x1, y1, x2, y2] = b.getAbsoluteBoundingBox(format=BBFormat.XYX2Y2)
        pred_bbox = y1, x1, y2, x2
        pred = {'pred_nr': b.getBboxID(), 'class_name': b.getClassId(), 'pred_score': b.getConfidence(),
                'bbox_orig_coords': pred_bbox}
        prediction_list.append(pred)
    prediction_dict['prediction_list'] = prediction_list
    return prediction_dict


def convert_table_structure_list_to_save_format(table_structure_annotations):
    prediction_dict = dict()

    prediction_dict['orig_img_shape'] = [None, None]

    prediction_list = []
    for b in table_structure_annotations:
        [x1, y1, w, h] = b['bbox']
        x2 = x1 + w
        y2 = y1 + h
        pred_bbox = [y1, x1, y2, x2]
        category = b['category']
        bbox_id = b['id']
        conf = b.get('score', 0.99)  # annotations generated in postprocessing have no score
        pred = {'pred_nr': bbox_id, 'class_name': category, 'pred_score': conf,
                'bbox_orig_coords': pred_bbox}
        prediction_list.append(pred)
    prediction_dict['prediction_list'] = prediction_list
    return prediction_dict


def evaluate_icdar_xmls(detections_dir, gt_source_dir, eval_jar_path):
    all_pdf_files_by_doc = {x.split('.pdf')[0]: x for x in os.listdir(gt_source_dir) if x.endswith('pdf')}

    special_xmls = defaultdict(list)
    for doc_id in all_pdf_files_by_doc.keys():
        gt_xml_path = os.path.join(gt_source_dir, doc_id + '-str.xml')
        if not os.path.isfile(gt_xml_path):
            a_path = gt_xml_path.replace('-str.xml', 'a-str.xml')
            if os.path.isfile(a_path):
                special_xmls[doc_id].append(a_path)
            b_path = gt_xml_path.replace('-str.xml', 'b-str.xml')
            if os.path.isfile(b_path):
                special_xmls[doc_id].append(b_path)

    prec_per_doc = dict()
    rec_per_doc = dict()
    f1_per_doc = dict()
    special_doc_id_mapping = dict()
    total_docs_number = len(all_pdf_files_by_doc)
    for doc_id in all_pdf_files_by_doc.keys():
        gt_pdf_path = os.path.join(gt_source_dir, doc_id + '.pdf')
        pred_xml_path = os.path.join(detections_dir, doc_id + '-str-gt_output.xml')
        if doc_id in special_xmls:
            gt_xml_paths = special_xmls[doc_id]
            special_doc_ids = [os.path.basename(x).split('-str.xml')[0] for x in gt_xml_paths]
            special_doc_id_mapping[doc_id] = special_doc_ids
        else:
            gt_xml_path = os.path.join(gt_source_dir, doc_id + '-str.xml')
            gt_xml_paths = [gt_xml_path]
            special_doc_ids = [doc_id]

        for current_doc_id, gt_xml_path in zip(special_doc_ids, gt_xml_paths):
            jaroutput = subprocess.check_output(
                ['java', '-jar', eval_jar_path, '-str', gt_xml_path, pred_xml_path, gt_pdf_path], encoding='utf-8',
                stderr=subprocess.DEVNULL)
            jaroutput = jaroutput.lower()
            pattern = re.compile("(table\s+[0-9]:\s+gt\ssize:)")
            matches = re.finditer(pattern, jaroutput)
            logger.debug('jar output: {}'.format(jaroutput))
            table_info_start_indeces = []
            for match in matches:
                start_index = match.span()[0]
                table_info_start_indeces.append(start_index)
            logger.debug("found {} individual tables".format(len(table_info_start_indeces)))
            table_infos = []
            for i, table_info_start_index in enumerate(table_info_start_indeces):
                if i == len(table_info_start_indeces) - 1:
                    table_info = jaroutput[table_info_start_index:]
                else:
                    table_info = jaroutput[table_info_start_index:table_info_start_indeces[i + 1]]
                table_infos.append(table_info.strip())

            total_TP = 0
            total_FP = 0
            total_FN = 0
            for table_info in table_infos:
                all_infos_found = re.findall(
                    "table ([0-9]):  gt size: ([0-9]+) corrdet: ([0-9]+) detected: ([0-9]+)  precision: [0-9]+ \/ [0-9]+ = ([0-9].[0-9]+|[a-z]+)  recall: [0-9]+ \/ [0-9]+ = ([0-9].[0-9]+)",
                    table_info)
                all_infos = all_infos_found[0]
                logger.debug("all infos for current table: {}".format(all_infos))
                table_nr = all_infos[0]
                gt_size = all_infos[1]
                correct_detected = all_infos[2]
                detected_total = all_infos[3]
                script_precision = all_infos[4]
                script_recall = all_infos[5]

                TP = int(correct_detected)
                FP = int(detected_total) - TP
                FN = int(gt_size) - int(correct_detected)
                if TP + FP == 0:
                    prec = 0
                else:
                    prec = TP / (TP + FP)
                rec = TP / (TP + FN)

                total_TP += TP
                total_FP += FP
                total_FN += FN

            if total_TP + total_FP == 0:
                total_prec = 0
            else:
                total_prec = total_TP / (total_TP + total_FP)
            total_rec = total_TP / (total_TP + total_FN)
            logger.debug(
                'got precision and recall for full document: prec: {}, recall: {}'.format(total_prec, total_rec))

            try:
                total_f1_score = (2 * total_prec * total_rec) / (total_prec + total_rec)
            except ZeroDivisionError:
                logger.debug('zero division for doc: {}'.format(doc_id))
                total_f1_score = 0

            prec_per_doc[current_doc_id] = total_prec
            rec_per_doc[current_doc_id] = total_rec
            f1_per_doc[current_doc_id] = total_f1_score

    # choose better gt score, if there were two available
    keep_best_doc_if_multiple_gt_versions_exist(special_doc_id_mapping, prec_per_doc, rec_per_doc, f1_per_doc)

    # calculate final f1 score according to ICDAR procedure (ICDAR 2013 Table Competition, V.D. "Combining Results"
    prec_avg = sum(prec_per_doc.values()) / float(len(prec_per_doc))
    recall_avg = sum(rec_per_doc.values()) / float(len(rec_per_doc))
    assert total_docs_number == len(rec_per_doc)
    assert total_docs_number == len(prec_per_doc)
    f1_avg = sum(f1_per_doc.values()) / float(len(f1_per_doc))
    try:
        f_score = (2 * prec_avg * recall_avg) / (prec_avg + recall_avg)
        logger.debug("difference between f1 avg and f1 from prec/recall: {}/{}".format(f1_avg, f_score))
    except ZeroDivisionError:
        logger.debug("zero division for total f1 score in {}".format(detections_dir))
        f_score = 0
    results_dict = {'precisions': prec_per_doc, 'recalls': rec_per_doc, 'f1s': f1_per_doc, 'avg_precision': prec_avg,
                    'avg_recall': recall_avg, 'f1': f_score, 'f1_alternative': f1_avg}
    logger.debug(
        '{}; {} total documents considered; avg precision: {}, avg recall: {}, avg F1: {}'.format(detections_dir,
                                                                                                  len(prec_per_doc),
                                                                                                  prec_avg, recall_avg,
                                                                                                  f_score))
    return results_dict


def keep_best_doc_if_multiple_gt_versions_exist(special_doc_id_mapping, prec_per_doc, rec_per_doc, f1_per_doc):
    for orig_doc_id, special_doc_ids in special_doc_id_mapping.items():
        f1_score_per_doc_id = []
        for special_doc_id in special_doc_ids:
            special_doc_f1 = f1_per_doc[special_doc_id]
            f1_score_per_doc_id.append(special_doc_f1)
        best_doc = f1_score_per_doc_id.index(max(f1_score_per_doc_id))
        prec_per_doc[orig_doc_id] = prec_per_doc[special_doc_ids[best_doc]]
        rec_per_doc[orig_doc_id] = rec_per_doc[special_doc_ids[best_doc]]
        f1_per_doc[orig_doc_id] = f1_per_doc[special_doc_ids[best_doc]]

        logger.debug('added best values prec/recall: {}/{} for key {}'.format(prec_per_doc[orig_doc_id],
                                                                              rec_per_doc[orig_doc_id], orig_doc_id))
        for special_doc_id in special_doc_ids:
            logger.debug('removing {} result from dicts.. ({} remains)'.format(special_doc_id, orig_doc_id))
            prec_per_doc.pop(special_doc_id)
            rec_per_doc.pop(special_doc_id)
            f1_per_doc.pop(special_doc_id)
