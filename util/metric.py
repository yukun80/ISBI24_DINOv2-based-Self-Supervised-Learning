"""
Metrics for computing evalutation results
Modified from vanilla PANet code by Wang et al.
"""

import numpy as np

class Metric(object):
    """
    Compute evaluation result

    Args:
        max_label:
            max label index in the data (0 denoting background)
        n_scans:
            number of test scans
    """
    def __init__(self, max_label=20, n_scans=None):
        self.labels = list(range(max_label + 1))  # all class labels
        self.n_scans = 1 if n_scans is None else n_scans

        # list of list of array, each array save the TP/FP/FN statistic of a testing sample
        self.tp_lst = [[] for _ in range(self.n_scans)]
        self.fp_lst = [[] for _ in range(self.n_scans)]
        self.fn_lst = [[] for _ in range(self.n_scans)]
        self.tn_lst = [[] for _ in range(self.n_scans)]
        self.slice_counter = [0 for _ in range(self.n_scans)]

    def reset(self):
        """
        Reset accumulated evaluation. 
        """
        # assert self.n_scans == 1, 'Should not reset accumulated result when we are not doing one-time batch-wise validation'
        del self.tp_lst, self.fp_lst, self.fn_lst, self.tn_lst
        self.tp_lst = [[] for _ in range(self.n_scans)]
        self.fp_lst = [[] for _ in range(self.n_scans)]
        self.fn_lst = [[] for _ in range(self.n_scans)]
        self.tn_lst = [[] for _ in range(self.n_scans)]
        self.slice_counter = [0 for _ in range(self.n_scans)]
        
    def reset_scan(self, n_scan, labels:list=None):
        """
        Reset accumulated evaluation for a specific scan. 
        """
        if labels is None:
            labels = self.labels
        for slice_idx in range(len(self.tp_lst[n_scan])):
            for label in labels:
                self.tp_lst[n_scan][slice_idx][label] = np.nan
                self.fp_lst[n_scan][slice_idx][label] = np.nan
                self.fn_lst[n_scan][slice_idx][label] = np.nan

    def record(self, pred, target, labels=None, n_scan=None):
        """
        Record the evaluation result for each sample and each class label, including:
            True Positive, False Positive, False Negative

        Args:
            pred:
                predicted mask array, expected shape is H x W
            target:
                target mask array, expected shape is H x W
            labels:
                only count specific label, used when knowing all possible labels in advance
        """
        assert pred.shape == target.shape

        if self.n_scans == 1:
            n_scan = 0

        # array to save the TP/FP/FN statistic for each class (plus BG)
        tp_arr = np.full(len(self.labels), np.nan)
        fp_arr = np.full(len(self.labels), np.nan)
        fn_arr = np.full(len(self.labels), np.nan)
        tn_arr = np.full(len(self.labels), np.nan)
        if labels is None:
            labels = self.labels
        else:
            labels = [0,] + labels

        for j, label in enumerate(labels):
            # Get the location of the pixels that are predicted as class j
            # idx = np.where(np.logical_and(pred == j, target != 255))
            # pred_idx_j = set(zip(idx[0].tolist(), idx[1].tolist()))
            # # Get the location of the pixels that are class j in ground truth
            # idx = np.where(target == j)
            # target_idx_j = set(zip(idx[0].tolist(), idx[1].tolist()))

            # # this should not work: if target_idx_j:  # if ground-truth contains this class
            # # the author is adding posion to the code
            # tp_arr[label] = len(set.intersection(pred_idx_j, target_idx_j))
            # fp_arr[label] = len(pred_idx_j - target_idx_j)
            # fn_arr[label] = len(target_idx_j - pred_idx_j)
            
            # calc the tp, fp and fn normally and compare the 2 values
            tp = ((pred == j).astype(int) * (target == j).astype(int)).sum()
            fp = ((pred == j).astype(int) * (target != j).astype(int)).sum()
            fn = ((pred != j).astype(int) * (target == j).astype(int)).sum()
            
            tn = ((pred != j).astype(int) * (target != j).astype(int)).sum()

            tp_arr[label] = tp
            fp_arr[label] = fp
            fn_arr[label] = fn
            tn_arr[label] = tn
            
            # assert tp == tp_arr[label]
            # assert fp == fp_arr[label]
            # assert fn == fn_arr[label]

        self.tp_lst[n_scan].append(tp_arr)
        self.fp_lst[n_scan].append(fp_arr)
        self.fn_lst[n_scan].append(fn_arr)
        self.tn_lst[n_scan].append(tn_arr)
        self.slice_counter[n_scan] += 1

    def _aggregate_stat(self, stat_lst, labels, n_scan=None):
        """Helper to aggregate statistics for selected labels."""
        if labels is None:
            labels = self.labels
        if isinstance(labels, int):
            labels = [labels]
        labels = list(dict.fromkeys(labels))  # ensure unique order preserved

        if n_scan is None:
            aggregated = []
            for _scan in range(self.n_scans):
                if not stat_lst[_scan]:
                    aggregated.append(np.zeros(len(labels)))
                    continue
                stacked = np.vstack(stat_lst[_scan])
                aggregated.append(np.nansum(stacked, axis=0).take(labels))
            return aggregated, labels

        if not stat_lst[n_scan]:
            return np.zeros(len(labels)), labels
        stacked = np.vstack(stat_lst[n_scan])
        return np.nansum(stacked, axis=0).take(labels), labels

    @staticmethod
    def _safe_divide(numerator, denominator):
        """Element-wise division guarding against zero denominators."""
        denominator = np.asarray(denominator, dtype=np.float64)
        numerator = np.asarray(numerator, dtype=np.float64)
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.true_divide(numerator, denominator)
            result[~np.isfinite(result)] = 0.0
        return result

    def _gather_stats(self, labels=None, n_scan=None):
        """
        Aggregate TP/FP/FN/TN statistics for the requested labels.
        Returns per-scan lists if n_scan is None, otherwise single arrays.
        """
        tp, label_idx = self._aggregate_stat(self.tp_lst, labels, n_scan)
        fp, _ = self._aggregate_stat(self.fp_lst, label_idx, n_scan)
        fn, _ = self._aggregate_stat(self.fn_lst, label_idx, n_scan)
        tn, _ = self._aggregate_stat(self.tn_lst, label_idx, n_scan)
        return tp, fp, fn, tn, label_idx

    def get_mIoU(self, labels=None, n_scan=None):
        """
        Compute mean IoU

        Args:
            labels:
                specify a subset of labels to compute mean IoU, default is using all classes
        """
        tp_sum, fp_sum, fn_sum, _, label_idx = self._gather_stats(labels, n_scan)
        # Sum TP, FP, FN statistic of all samples
        if n_scan is None:
            if not tp_sum:
                return (np.zeros(len(label_idx)), np.zeros(len(label_idx)), 0.0, 0.0)
            mIoU_class = []
            for scan_tp, scan_fp, scan_fn in zip(tp_sum, fp_sum, fn_sum):
                denom = scan_tp + scan_fp + scan_fn
                mIoU_class.append(self._safe_divide(scan_tp, denom))
            mIoU_class = np.vstack(mIoU_class)
            mIoU = mIoU_class.mean(axis=1)

            return (mIoU_class.mean(axis=0), mIoU_class.std(axis=0),
                    mIoU.mean(axis=0), mIoU.std(axis=0))
        else:
            denom = tp_sum + fp_sum + fn_sum
            mIoU_class = self._safe_divide(tp_sum, denom)
            mIoU = float(mIoU_class.mean()) if mIoU_class.size else 0.0

            return mIoU_class, mIoU

    def get_mDice(self, labels=None, n_scan=None, give_raw = False):
        """
        Compute mean Dice score (in 3D scan level)

        Args:
            labels:
                specify a subset of labels to compute mean IoU, default is using all classes
        """
        # NOTE: unverified
        tp_sum, fp_sum, fn_sum, _, label_idx = self._gather_stats(labels, n_scan)
        if n_scan is None:
            if not tp_sum:
                empty = np.zeros(len(label_idx))
                if not give_raw:
                    return (empty, empty, 0.0, 0.0)
                return (empty, empty, 0.0, 0.0, np.zeros((0, len(label_idx))))
            mDice_class = []
            for scan_tp, scan_fp, scan_fn in zip(tp_sum, fp_sum, fn_sum):
                denom = 2 * scan_tp + scan_fp + scan_fn
                mDice_class.append(self._safe_divide(2 * scan_tp, denom))
            mDice_class = np.vstack(mDice_class)
            mDice = mDice_class.mean(axis=1)
            if not give_raw:
                return (mDice_class.mean(axis=0), mDice_class.std(axis=0),
                    mDice.mean(axis=0), mDice.std(axis=0))
            else:
                return (mDice_class.mean(axis=0), mDice_class.std(axis=0),
                    mDice.mean(axis=0), mDice.std(axis=0), mDice_class)

        else:
            denom = 2 * tp_sum + fp_sum + fn_sum
            mDice_class = self._safe_divide(2 * tp_sum, denom)
            mDice = float(mDice_class.mean()) if mDice_class.size else 0.0

            if not give_raw:
                return (mDice_class, mDice, mDice_class)
            
            return (mDice_class, mDice, mDice_class)

    def get_mPrecRecall(self, labels=None, n_scan=None, give_raw = False):
        """
        Compute precision and recall

        Args:
            labels:
                specify a subset of labels to compute mean IoU, default is using all classes
        """
        # NOTE: unverified
        tp_sum, fp_sum, fn_sum, _, label_idx = self._gather_stats(labels, n_scan)
        if n_scan is None:
            if not tp_sum:
                zeros = np.zeros(len(label_idx))
                if not give_raw:
                    return (zeros, zeros, 0.0, 0.0, zeros, zeros, 0.0, 0.0)
                return (zeros, zeros, 0.0, 0.0, zeros, zeros, 0.0, 0.0, np.zeros((0, len(label_idx))), np.zeros((0, len(label_idx))))
            mPrec_class = []
            mRec_class = []
            for scan_tp, scan_fp, scan_fn in zip(tp_sum, fp_sum, fn_sum):
                prec = self._safe_divide(scan_tp, scan_tp + scan_fp)
                rec = self._safe_divide(scan_tp, scan_tp + scan_fn)
                mPrec_class.append(prec)
                mRec_class.append(rec)
            mPrec_class = np.vstack(mPrec_class)
            mRec_class = np.vstack(mRec_class)

            mPrec = mPrec_class.mean(axis=1)
            mRec  = mRec_class.mean(axis=1)
            if not give_raw:
                return (mPrec_class.mean(axis=0), mPrec_class.std(axis=0), mPrec.mean(axis=0), mPrec.std(axis=0), mRec_class.mean(axis=0), mRec_class.std(axis=0), mRec.mean(axis=0), mRec.std(axis=0))
            else:
                return (mPrec_class.mean(axis=0), mPrec_class.std(axis=0), mPrec.mean(axis=0), mPrec.std(axis=0), mRec_class.mean(axis=0), mRec_class.std(axis=0), mRec.mean(axis=0), mRec.std(axis=0), mPrec_class, mRec_class)


        else:
            mPrec_class = self._safe_divide(tp_sum, tp_sum + fp_sum)
            mRec_class = self._safe_divide(tp_sum, tp_sum + fn_sum)
            mPrec = float(mPrec_class.mean()) if mPrec_class.size else 0.0
            mRec = float(mRec_class.mean()) if mRec_class.size else 0.0

            return mPrec_class, None, mPrec, None, mRec_class, None, mRec, None,  mPrec_class, mRec_class

    def get_precision_recall_f1(self, labels=None, n_scan=None, give_raw=False):
        """
        Compute precision, recall, and F1-score (Dice) for selected labels.
        """
        tp_sum, fp_sum, fn_sum, _, label_idx = self._gather_stats(labels, n_scan)
        if n_scan is None:
            if not tp_sum:
                zeros = np.zeros(len(label_idx))
                if not give_raw:
                    return {
                        'precision': (zeros, zeros, 0.0, 0.0),
                        'recall': (zeros, zeros, 0.0, 0.0),
                        'f1': (zeros, zeros, 0.0, 0.0),
                    }
                return {
                    'precision': (zeros, zeros, 0.0, 0.0, np.zeros((0, len(label_idx)))),
                    'recall': (zeros, zeros, 0.0, 0.0, np.zeros((0, len(label_idx)))),
                    'f1': (zeros, zeros, 0.0, 0.0, np.zeros((0, len(label_idx)))),
                }

            prec_stack = []
            rec_stack = []
            f1_stack = []
            for scan_tp, scan_fp, scan_fn in zip(tp_sum, fp_sum, fn_sum):
                prec = self._safe_divide(scan_tp, scan_tp + scan_fp)
                rec = self._safe_divide(scan_tp, scan_tp + scan_fn)
                f1 = self._safe_divide(2 * prec * rec, prec + rec)
                prec_stack.append(prec)
                rec_stack.append(rec)
                f1_stack.append(f1)

            prec_stack = np.vstack(prec_stack)
            rec_stack = np.vstack(rec_stack)
            f1_stack = np.vstack(f1_stack)

            precision_stats = (prec_stack.mean(axis=0), prec_stack.std(axis=0),
                               prec_stack.mean(axis=1).mean(), prec_stack.mean(axis=1).std())
            recall_stats = (rec_stack.mean(axis=0), rec_stack.std(axis=0),
                            rec_stack.mean(axis=1).mean(), rec_stack.mean(axis=1).std())
            f1_stats = (f1_stack.mean(axis=0), f1_stack.std(axis=0),
                        f1_stack.mean(axis=1).mean(), f1_stack.mean(axis=1).std())
            result = {
                'precision': precision_stats,
                'recall': recall_stats,
                'f1': f1_stats,
            }
            if give_raw:
                result['precision'] = result['precision'] + (prec_stack,)
                result['recall'] = result['recall'] + (rec_stack,)
                result['f1'] = result['f1'] + (f1_stack,)
            return result

        # single scan aggregation
        precision = self._safe_divide(tp_sum, tp_sum + fp_sum)
        recall = self._safe_divide(tp_sum, tp_sum + fn_sum)
        f1 = self._safe_divide(2 * precision * recall, precision + recall)
        precision_stats = (precision, None, float(precision.mean()) if precision.size else 0.0, None)
        recall_stats = (recall, None, float(recall.mean()) if recall.size else 0.0, None)
        f1_stats = (f1, None, float(f1.mean()) if f1.size else 0.0, None)
        result = {
            'precision': precision_stats,
            'recall': recall_stats,
            'f1': f1_stats,
        }
        if give_raw:
            result['precision'] = result['precision'] + (precision[np.newaxis, ...],)
            result['recall'] = result['recall'] + (recall[np.newaxis, ...],)
            result['f1'] = result['f1'] + (f1[np.newaxis, ...],)
        return result

    def get_overall_accuracy(self, labels=None, n_scan=None, give_raw=False):
        """
        Compute overall pixel accuracy (TP+TN / Total) for selected labels.
        """
        tp_sum, fp_sum, fn_sum, tn_sum, label_idx = self._gather_stats(labels, n_scan)
        if n_scan is None:
            if not tp_sum:
                zeros = np.zeros(len(label_idx))
                if not give_raw:
                    return zeros, zeros, 0.0, 0.0
                return zeros, zeros, 0.0, 0.0, np.zeros((0, len(label_idx)))

            acc_class_stack = []
            overall_stack = []
            for scan_tp, scan_fp, scan_fn, scan_tn in zip(tp_sum, fp_sum, fn_sum, tn_sum):
                total = scan_tp + scan_fp + scan_fn + scan_tn
                per_class = self._safe_divide(scan_tp + scan_tn, total)
                acc_class_stack.append(per_class)
                total_sum = total.sum()
                overall_sum = (scan_tp + scan_tn).sum()
                overall_stack.append(overall_sum / total_sum if total_sum > 0 else 0.0)

            acc_class_stack = np.vstack(acc_class_stack)
            overall_stack = np.asarray(overall_stack, dtype=np.float64)

            if give_raw:
                return (acc_class_stack.mean(axis=0), acc_class_stack.std(axis=0),
                        overall_stack.mean(), overall_stack.std(), acc_class_stack)

            return (acc_class_stack.mean(axis=0), acc_class_stack.std(axis=0),
                    overall_stack.mean(), overall_stack.std())

        total = tp_sum + fp_sum + fn_sum + tn_sum
        per_class = self._safe_divide(tp_sum + tn_sum, total)
        total_sum = total.sum()
        overall_sum = (tp_sum + tn_sum).sum()
        overall = overall_sum / total_sum if total_sum > 0 else 0.0
        if give_raw:
            return per_class, None, overall, None, per_class[np.newaxis, ...]
        return per_class, None, overall, None

    def get_mIoU_binary(self, n_scan=None):
        """
        Compute mean IoU for binary scenario
        (sum all foreground classes as one class)
        """
        # Sum TP, FP, FN statistic of all samples
        if n_scan is None:
            tp_sum = [np.nansum(np.vstack(self.tp_lst[_scan]), axis=0)
                      for _scan in range(self.n_scans)]
            fp_sum = [np.nansum(np.vstack(self.fp_lst[_scan]), axis=0)
                      for _scan in range(self.n_scans)]
            fn_sum = [np.nansum(np.vstack(self.fn_lst[_scan]), axis=0)
                      for _scan in range(self.n_scans)]

            # Sum over all foreground classes
            tp_sum = [np.c_[tp_sum[_scan][0], np.nansum(tp_sum[_scan][1:])]
                      for _scan in range(self.n_scans)]
            fp_sum = [np.c_[fp_sum[_scan][0], np.nansum(fp_sum[_scan][1:])]
                      for _scan in range(self.n_scans)]
            fn_sum = [np.c_[fn_sum[_scan][0], np.nansum(fn_sum[_scan][1:])]
                      for _scan in range(self.n_scans)]

            # Compute mean IoU classwisely and average across classes
            mIoU_class = np.vstack([tp_sum[_scan] / (tp_sum[_scan] + fp_sum[_scan] + fn_sum[_scan])
                                    for _scan in range(self.n_scans)])
            mIoU = mIoU_class.mean(axis=1)

            return (mIoU_class.mean(axis=0), mIoU_class.std(axis=0),
                    mIoU.mean(axis=0), mIoU.std(axis=0))
        else:
            tp_sum = np.nansum(np.vstack(self.tp_lst[n_scan]), axis=0)
            fp_sum = np.nansum(np.vstack(self.fp_lst[n_scan]), axis=0)
            fn_sum = np.nansum(np.vstack(self.fn_lst[n_scan]), axis=0)

            # Sum over all foreground classes
            tp_sum = np.c_[tp_sum[0], np.nansum(tp_sum[1:])]
            fp_sum = np.c_[fp_sum[0], np.nansum(fp_sum[1:])]
            fn_sum = np.c_[fn_sum[0], np.nansum(fn_sum[1:])]

            mIoU_class = tp_sum / (tp_sum + fp_sum + fn_sum)
            mIoU = mIoU_class.mean()

            return mIoU_class, mIoU
