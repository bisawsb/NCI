MaxMRRRank = 100

def recall(args):
    recall_avg = None
    q_gt, q_pred = {}, {}
    with open(args.res1_save_path, "r") as f:
        prev_q = ""
        for line in f.readlines():
            query, pred, gt, rank = line[:-1].split("\t")
            if query != prev_q:
                q_pred[query] = pred.split(",")
                q_pred[query] = q_pred[query]
                prev_q = query
            if query in q_gt:
                if len(q_gt[query]) <= 100:
                    q_gt[query].add(gt)
            else:
                q_gt[query] = set()
                q_gt[query].add(gt)

    for i in args.recall_num:
        total = 0
        for q in q_pred:
            right = 0
            wrong = 0
            for p in q_gt[q]:
                if p in q_pred[q][:i]:
                    right += 1
                else:
                    wrong += 1
            recall = right / (right + wrong)
            total += recall
        recall_avg = total / len(q_pred)
        print(f"recall@{i}: {recall_avg}")
        print('-------------------------')
    return recall_avg


def MRR100(args):
    mrr_total = 0
    query_num = 0
    with open(args.res1_save_path, "r") as f:
        for line in f.readlines():
            query, pred, gt, rank = line.split("\t")
            pred_list = pred.split(",")
            # pred_list = pred_list[:10]
            if gt in pred_list:
                rank = pred_list.index(gt) + 1
                mrr_total += 1 / rank
            else:
                mrr_total += 1 / len(pred_list)
            query_num += 1

    mrr = mrr_total / query_num
    print('{}: {}'.format('MRR100', mrr))
    return mrr
