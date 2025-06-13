import torch
import re
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer
from frozen_fusion_model import Frozen_Fusion
from utils import cal_rouge, getRouge2
from load_data import loadClusterData
from run_functions import extractive_infer
from underthesea import word_tokenize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
abstract_tokenizer_path = "vinai/bartpho-syllable-base"
abstract_model_path = "./checkpoints/checkpoint-2200"

stopword_path = "./VLSP Dataset/vietnamese-stopwords-dash.txt"
extractive_model_path = "./checkpoints/graph_frozen_fusion.pt"
model = Frozen_Fusion(input=768).to(device)
model.load_state_dict(torch.load(extractive_model_path, map_location=device), strict=True)

phobert = AutoModel.from_pretrained("vinai/phobert-base-v2").to(device)
phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
model_summarization = AutoModelForSeq2SeqLM.from_pretrained(abstract_model_path).to(device)
tokenizer_summarization = AutoTokenizer.from_pretrained(abstract_tokenizer_path)


def normalize_text(text):
    text = str(text).replace('_', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([.,;:?)/!?”])', r'\1', text)
    text = re.sub(r'([\(“])\s+', r'\1', text)
    return text

def track_changes(old_words, new_words):
    # Find the longest common subsequence (LCS) between the two word sequences
    def get_lcs_matrix(words1, words2):
        m, n = len(words1), len(words2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if words1[i-1] == words2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp

    def get_lcs(words1, words2, dp):
        i, j = len(words1), len(words2)
        lcs = []

        while i > 0 and j > 0:
            if words1[i-1] == words2[j-1]:
                lcs.append((i-1, j-1))
                i -= 1
                j -= 1
            elif dp[i-1][j] > dp[i][j-1]:
                i -= 1
            else:
                j -= 1

        return sorted(lcs)

    # Find the changed segments at word level
    dp_matrix = get_lcs_matrix(old_words, new_words)
    lcs_positions = get_lcs(old_words, new_words, dp_matrix)

    changes = []
    old_pos = 0
    new_pos = 0

    # Process matching and non-matching segments
    for old_idx, new_idx in lcs_positions:
        # If there's a gap before this match, it's a change
        if old_idx > old_pos or new_idx > new_pos:
            changes.append((old_pos, old_idx, new_pos, new_idx))

        # Move positions after the match
        old_pos = old_idx + 1
        new_pos = new_idx + 1

    # Check if there's a change at the end
    if old_pos < len(old_words) or new_pos < len(new_words):
        changes.append((old_pos, len(old_words), new_pos, len(new_words)))

    return changes


class Abstractive_Summarization:
    @staticmethod
    def generateSummaryBySent(texts, batch=32):
        model_summarization.eval()
        predictions = []
        with torch.no_grad():
            for i in range(0, len(texts), batch):
                batch_texts = texts[i:i+batch]
                inputs = tokenizer_summarization(batch_texts, padding=True, max_length=1024, truncation=True,
                                                return_tensors='pt').to(device)
                outputs = model_summarization.generate(**inputs, num_beams=5,
                                                    early_stopping=True, no_repeat_ngram_size=3)
                prediction = tokenizer_summarization.batch_decode(outputs, skip_special_tokens=True)
                predictions.extend(prediction)
        return predictions


def get_summary(scores, sents, max_sent=5):
    ranked_score_idxs = torch.argsort(scores, dim=0, descending=True)
    sents = [s.replace('_', ' ') for s in sents]
    summSentIDList = []
    for i in ranked_score_idxs:
        if len(summSentIDList) >= max_sent: break
        s = sents[i]

        replicated, delIDs = False, []
        for chosedID in summSentIDList:
            if getRouge2(s, sents[chosedID], 'p') >= 0.45:
                delIDs.append(chosedID)
            if getRouge2(sents[chosedID], s, 'p') >= 0.45:
                replicated = True
                break
        if replicated: continue

        for delID in delIDs:
            del summSentIDList[summSentIDList.index(delID)]
        summSentIDList.append(i)
    summSentIDList = sorted(summSentIDList)
    return [s for i, s in enumerate(sents) if i in summSentIDList]


def MultiDocSummarizationAPI(texts, compress_ratio, golden_abs=None):
    """
    Summarizes a list of documents using both extractive and abstractive methods.

    Parameters:
    - texts (list of str): A list of document texts to be summarized.
    - compress_ratio (float): A ratio or count determining the number of sentences in the summary.
      If less than 1, it represents the fraction of the original sentences to include in the summary.
      If 1 or greater, it represents the exact number of sentences to include in the summary.

    Returns:
    - dict: A dictionary containing:
        - 'extractive_summ' (str): The extractive summary of the documents.
        - 'abstractive_summ' (str): The abstractive summary of the documents.
    """
    assert compress_ratio > 0, "Compress ratio need to be greater than 0."
    docs = [text.strip() for text in texts]
    data_tree = loadClusterData(docs)
    scores, sents = extractive_infer(model, data_tree)

    output_sent_cnt = int(len(sents) * compress_ratio) if compress_ratio < 1 else int(compress_ratio)
    print('Expected sentence count:', output_sent_cnt)

    extractive_summ_sents = [normalize_text(sent) for sent in get_summary(scores, sents, max_sent=output_sent_cnt)]
    extractive_summ = ' '.join(extractive_summ_sents)

    abstractive_summ_sents = Abstractive_Summarization.generateSummaryBySent(extractive_summ_sents)
    abstractive_summ_sents = [normalize_text(s) for s in abstractive_summ_sents]
    final_sents = []
    for ii, (ext, abs) in enumerate(zip(extractive_summ_sents, abstractive_summ_sents)):
        if ii == 0:
            final_sents.append(ext)
            continue
        abs_splits, ext_splits = word_tokenize(abs), word_tokenize(ext)
        abs_splits_cop, ext_splits_cop = abs_splits.copy(), ext_splits.copy()
        if len(abs_splits_cop):
            abs_splits_cop[-1] = abs_splits[-1][:-1] if len(abs_splits[-1]) and abs_splits[-1][-1] == '.' else abs_splits[-1]
        if len(ext_splits_cop):
            ext_splits_cop[-1] = ext_splits[-1][:-1] if len(ext_splits[-1]) and ext_splits[-1][-1] == '.' else ext_splits[-1]

        changes, abs_parts = track_changes(ext_splits_cop, abs_splits_cop), [(0, len(abs_splits))]
        for start_old, end_old, start_new, end_new in changes:
            old_part = ' '.join(ext_splits[start_old:end_old])
            # Revert change in the cases of spelling errors
            revert, ignoreFirstSentWord = False, 1 if start_old == 0 else 0
            old_names = {}
            for w in ext_splits_cop[start_old + ignoreFirstSentWord:end_old]:
                if len(w) == 0: continue
                if 'A'<=w[0]<='Z' or w[0] in ['Ă', 'Â', 'Đ', 'Ê', 'Ô', 'Ơ', 'Ư']:
                    if w in old_names:
                        old_names[w] += 1
                    else:
                        old_names[w] = 1

            for w in abs_splits_cop[start_new + ignoreFirstSentWord:end_new]:
                if len(w) == 0: continue
                if 'A'<=w[0]<='Z' or w[0] in ['Ă', 'Â', 'Đ', 'Ê', 'Ô', 'Ơ', 'Ư']:
                    if w in old_names:
                        old_names[w] -= 1
                        if old_names[w] < 0:
                            revert = True
                            break
                    else:
                        revert = True
                        break
            if revert:
                pop_part = abs_parts[-1]
                abs_parts.pop()
                abs_parts.extend([(pop_part[0], start_new), old_part, (end_new, pop_part[1])])
                # print('\nOLD:', old_part, '\n', ' '.join(abs_splits[start_new:end_new]))
                # print(ext, '\n', abs)

        abs = ' '.join([part if isinstance(part, str) else ' '.join(abs_splits[part[0]:part[1]]) for part in abs_parts])
        final_sents.append(normalize_text(abs))
    abstract_summ = ' '.join(final_sents)

    score_ext = cal_rouge(golden_abs, extractive_summ) if golden_abs is not None else None
    score_abs = cal_rouge(golden_abs, abstract_summ) if golden_abs is not None else None
    return {'extractive_summ': extractive_summ,
            'abstractive_summ': abstract_summ,
            'score_ext': score_ext,
            'score_abs': score_abs}


def main():
    texts = [
        "Nghị quyết 54 sắp hết hiệu lực vào cuối năm 2022 nhưng hiện vẫn chưa có dự thảo Nghị quyết mới thay thế, dẫn đến tiềm ẩn nguy cơ trần nợ công sẽ bị hạ xuống đối với TP.HCM. Báo cáo Kinh tế vĩ mô TP.HCM do Trường Đại học Kinh tế - Luật (Đại học Quốc gia TP.HCM) và Viện Nghiên cứu phát triển Công nghệ ngân hàng ĐHQG-HCM thực hiện đã chỉ ra các thách thức dai dẳng đang cản trở sự phát triển của TP.HCM.\nTrong bản báo cáo dài 21 trang vừa công bố, nhóm nghiên cứu do TS. Phạm Thị Thanh Xuân chủ trì, nhận định, nợ công của TP vốn đã sát trần. Tốc độ tăng trưởng kinh tế giảm thấp hai năm qua khiến room nợ công càng hẹp hơn.\nTrần nợ công hạ, TP.HCM hết room tín dụng\nCụ thể, nợ công của thành phố chiếm tỷ lệ 70% thu ngân sách; sau đó được nới lên mức kịch trần là 90% theo Nghị quyết 54/2017/QH14. Tuy nhiên, vấn đề đặt ra là Nghị quyết 54 sắp hết hiệu lực vào cuối năm 2022 nhưng hiện vẫn chưa có dự thảo Nghị quyết mới thay thế, dẫn đến tiềm ẩn nguy cơ trần nợ công sẽ bị hạ xuống còn 70%.\nMặt khác, theo quan điểm chủ đạo trong Chiến lược nợ công đến năm 2030 của Chính phủ, các địa phương cần đảm bảo an toàn nợ công và giữ mức nợ công không quá 60% GDP. Đây chính là điểm nghẽn khiến TP.HCM hết room tín dụng, vướng mắc trong khâu huy động vốn vay để có thể triển khai hàng loạt chương trình, dự án đầu tư công trọng điểm cấp thiết sắp tới.\nKhi hết room tín dụng, TP.HCM sẽ gặp khó khi huy động vốn để triển khai hàng loạt chương trình, dự án đầu tư công trọng điểm (ảnh: Trần Chung)\nNhóm chuyên gia còn cho rằng, nguồn lực đặc thù từ \"cơ chế 54\" chưa kịp hiện thực đã gần về đích. Bởi, Nghị quyết số 54 có hiệu lực từ tháng 1/2018 đến hết năm 2022, như vậy chưa kịp phát huy hiệu quả mà thời hạn đã gần hết, một số nội dung triển khai vào thực tế chậm so với kế hoạch.\nĐơn cử, một trọng tâm của Nghị quyết 54 là “cơ chế tài chính đặc thù” chưa được phát huy như mong đợi. Các khoản thu đặc thù từ hạ tầng cảng biển mới được triển khai từ 1/4/2022 do phải hoãn hai lần vì dịch Covid-19, dù doanh thu mang lại còn khiêm tốn nhưng trở thành vấn đề trái chiều ở cấp vĩ mô. Chính sách thu phí về bản chất và mục tiêu tạo nguồn tài chính tại chỗ để nâng cấp cơ sở hạ tầng cảng biển, cả đường bộ lẫn đường thủy. Tuy nhiên, thời điểm tổ chức thu phí rơi vào giai đoạn nền kinh tế chưa phục hồi hoàn toàn. Do đó, khoản thu đã tạo thêm gánh nặng chi phí cho DN, gián tiếp hạn chế phần nào khả năng phục hồi của DN sau dịch.\nCòn khoản thu đặc thù từ quản lý tài sản công do cơ quan Trung ương quản lý, sử dụng trên địa bàn TP.HCM chưa hiện thực, chủ yếu do tính chất phức tạp cần sự phối hợp giữa các bộ ngành. Đến nay, chưa tạo được dòng tiền vào cho ngân sách cả Trung ương lẫn TP.HCM từ chính sách này.\nChính sách thu phí đỗ xe vỉa hè cũng thất bại, thậm chí phát sinh lỗ, nguyên nhân chủ yếu do ứng dụng công nghệ chưa hoàn thiện cũng như thiếu chế tài xử lý vi phạm.\nCó 347.000 tỷ mà chỉ giải ngân được 22.000 tỷ\nMột điểm nghẽn cũng ảnh hưởng tới TP.HCM là các gói kích thích kinh tế của quốc gia vẫn chưa được hấp thụ vào nền kinh tế. Gói 347.000 tỷ đồng hỗ trợ chương trình phục hồi và phát triển kinh tế - xã hội được Chính phủ trình Quốc hội nhằm khôi phục nhanh chuỗi sản xuất, tạo sức bật tăng trưởng tập trung thực hiện trong hai năm 2022-2023, song đến nay chưa giải ngân như kỳ vọng. Theo tính toán, con số giải ngân mới chỉ đạt 6%, tương đương 22.000 tỷ chảy vào nền kinh tế. Cần lưu ý, mục tiêu của các gói kích thích kinh tế phải ưu tiên và có tính khả thi ngay trong ngắn hạn.\nMột, ưu tiên giảm gánh nặng chi phí cho các DN, từ đó thúc đẩy sản xuất bật trở lại trong ngắn hạn. Mục tiêu này nên tập trung vào các công cụ giãn và giảm thuế, phí, giãn thuế liên quan đầu vào sản xuất. Các công cụ này không những hiệu quả mà còn tránh được áp lực lạm phát.\nHai, ưu tiên kích cầu tiêu thụ nội địa. Rõ ràng sức tăng trưởng kinh tế trong 6 tháng đầu năm có sự đóng góp rất lớn từ sức tăng của tổng cầu. Cầu nội địa phục hồi kéo sản xuất phục hồi tốt, lại ít chịu ảnh hưởng từ bên ngoài. Công cụ kích cầu hiệu quả vẫn là giảm các sắc thuế thu ở khâu tiêu dùng và hỗ trợ cho DN bình ổn giá, gia tăng chương trình khuyến mãi.\n“Đối sánh với hai mục tiêu này, dường như các gói kích thích kinh tế đã thiết kế phù hợp nhưng chưa thực sự thẩm thấu được vào nền kinh tế cả nước nói chung cũng như TP.HCM nói riêng, theo đúng tiến độ kỳ vọng”, báo cáo nêu.",
        "Sau gần 5 năm thực hiện các cơ chế, chính sách đặc thù theo NQ 54 của Quốc hội, tới nay TP.HCM đã đạt được một số kết quả nhất định như thông qua 32 dự án có chuyển mục đích sử dụng trên 10ha, quyết định chủ trương đầu tư 05 dự án nhóm A, thực hiện thu phí môi trường đối với nước thải công nghiệp… Tuy nhiên, thành phố chưa tận dụng được tối đa lợi thế của các cơ chế, chính sách mà NQ 54 mang lại. Theo đó, hơn 4 năm qua thành phố không được hưởng một đồng nào từ khoản thu tiền sử dụng đất khi bán các tài sản công của các đơn vị trung ương trên địa bàn. Việc cổ phần hóa các doanh nghiệp nhà nước cũng không thực hiện được do chưa có hướng dẫn cụ thể từ trung ương. Nhiều cơ chế, chính sách của NQ 54 chưa được tận dụng triệt để dẫn tới hiệu quả chưa cao.\nÔng PHAN VĂN MÃI, Chủ tịch UBND TP. Hồ Chí Minh: \"Phân cấp cho TP. HCM trong thời gian qua lúc đầu tiếp cận theo NQ 54, việc này phân cấp cho thành phố, nhưng khi đi làm phải hỏi ý kiến bộ, lại quay lại quy định của pháp luật nên rất khó. Vì vậy, phân cấp phải đi liền với việc giao các điều kiện để thực hiện. Đây là điều cốt lõi của nghị quyết thay thế Nghị quyết 54 trong thời gian tới.\"\nNhiều đại biểu bày tỏ lo ngại, sau khi Nghị quyết 54 kết thúc vào tháng 11 tới, các nội dung đang triển khai theo cơ chế này sẽ tiếp tục được duy trì thực hiện như thế nào, UBND thành phố cần làm rõ và có sự chuẩn bị chu đáo.\nÔng TĂNG HỮU PHONG, Đại biểu HĐND TP. Hồ Chí Minh: \"Hiện đang ở giai đoạn tổng kết Nghị quyết 54, vậy từ khi kết thúc hiệu lực tới khi có Nghị quyết mới thay thế thì chúng ta điều hành những nội dung của Nghị quyết 54 trên cơ sở pháp lý nào?\"\nÔng PHAN VĂN MÃI, Chủ tịch UBND TP. Hồ Chí Minh: \"Khi Chủ tịch Quốc hội vào làm việc với thành phố, chúng tôi cũng báo cáo lộ trình và Chủ tịch cũng ủng hộ, làm sao chúng ta kịp trình trong kỳ họp cuối năm, nếu vậy thì cũng liên tục, còn nếu có khoảng thời gian thì chúng ta sẽ có cơ chế để nối.\"\nĐại biểu đề nghị, ngay từ bây giờ TP. HCM cần gấp rút hoàn thiện báo cáo tổng kết tình hình thực hiện Nghị quyết 54 để trình QH đúng thời gian dự kiến, đồng thời cần định lượng chính xác các chỉ tiêu đạt được để có cái nhìn, giải pháp cụ thể làm cơ sở đề xuất, xây dựng Nghị quyết mới phù hợp hơn thay thế nghị quyết 54 trong thời gian tới.\nThực hiện : Thùy Vân Tăng Sắc",
        "Nhiều chuyên gia đề xuất cụ thể về cơ chế mới thay thế Nghị quyết 54 nhằm tăng quyền tự chủ cho TP.HCM. Chiều 13-7, ĐH Quốc gia TP.HCM tổ chức tọa đàm “Đề xuất và kiến nghị một số vấn đề triển khai Nghị quyết 54”. Các chuyên gia đã có nhiều góp ý cho nghị quyết mới thay thế Nghị quyết 54.\nCho phép nhưng chưa được quyền chủ động\nTS Thái Thị Tuyết Dung, Trường ĐH Kinh tế - Luật TP.HCM, cho rằng kết quả thực hiện Nghị quyết 54 chưa đạt như kỳ vọng. Theo bà, nguyên nhân là do nhiều nội dung trong nghị quyết chưa được phân cấp triệt để.\n“TP.HCM được cho phép nhưng chưa được quyền chủ động thực hiện, vẫn phải ra Trung ương xin thêm cơ chế” - bà nói.\nBà dẫn chứng, Nghị quyết 54 cho phép TP được hưởng 50% tiền bán đấu giá đất và tài sản trên đất của các cơ quan trung ương trên địa bàn TP. Nhưng thực tế không dễ bán tài sản công, phải qua quá nhiều thủ tục, phải được cấp thẩm quyền phê duyệt phương án.\nTP cũng chưa được tự chủ trong việc tổ chức bộ máy chính quyền đô thị, nhất là chủ động thành lập các đơn vị trực thuộc, các cơ quan chuyên môn thuộc UBND cấp tỉnh. Mối quan hệ giữa Nghị quyết 54 với các đạo luật chuyên ngành chưa tạo thành nguyên tắc thống nhất.\n“Khi có sự khác nhau giữa nghị quyết và luật, nhiều trường hợp nghị quyết không được ưu tiên áp dụng. Thay vì xin cơ chế giải quyết những vấn đề riêng lẻ, trao quyền nhỏ giọt, cần kiến nghị cho phép HĐND TP.HCM được quyền quyết định bộ máy chính quyền, nhất là thành lập, giải thể các cơ quan, đơn vị trực thuộc và quyết định biên chế công chức trong cơ quan của HĐND, UBND, đơn vị sự nghiệp công lập của UBND các cấp” - bà Dung gợi mở.\nCùng ý kiến, PGS-TS Nguyễn Anh Phong, Trường ĐH Kinh tế - Luật, đánh giá: Nghị quyết 54 để cho TP tăng quyền tự chủ, có ngân sách để hoạt động. Tuy nhiên, có những cái “cho cũng như không”. Đơn cử đối với thuế môi trường và thuế tiêu thụ đặc biệt, TP có cơ chế nhưng không dám tăng.\n“Ví như tăng thuế lên nhà máy sản xuất bia thì chắc chắn doanh nghiệp sẽ chuyển nhà máy tới các tỉnh lân cận” - ông Phong dẫn chứng và cho rằng cơ chế phải thật sự đặc thù và TP phải được hưởng đặc thù đó.\nTS Trương Minh Huy Vũ, Giám đốc Khu công nghệ phần mềm, ĐH Quốc gia TP.HCM, đề xuất cơ chế của TP.HCM phải bằng hoặc vượt trội các đô thị đang đóng vai trò trung tâm kinh tế quốc gia; trao quyền tương ứng với trách nhiệm.\n“TP được trao quyền lớn hơn, đồng nghĩa cam kết phải đóng góp cho cả nước; không dàn trải mà chỉ tập trung vào các lĩnh vực ưu tiên, gắn với việc thúc đẩy những thế mạnh của TP về thương mại, dịch vụ, tài chính, công nghệ, đổi mới, sáng tạo, thu hút nhà đầu tư lớn” - TS Vũ nêu ý kiến.\nGS-TS Nguyễn Kỳ Phùng, Phó Chủ tịch UBND TP Thủ Đức, cho rằng khi TP.HCM làm việc với các bộ, ngành về nghị quyết mới cần làm rõ, thống nhất các nội dung phân cấp trước khi ban hành. “Cái gì phân cấp được thì phân cấp luôn, không để ra đời rồi lại chạy đi xin từng cái, không biết mất bao nhiêu thời gian” - ông đề xuất và cho rằng phân cấp chính quyền đô thị cần dựa trên ba khía cạnh chính trị, hành chính và tài khóa.\nChú trọng vào nguồn lực cán bộ\nPGS-TS Trần Ngọc Anh cho rằng để đề xuất cơ chế, chính sách đặc thù, TP.HCM phải trình bày với Trung ương trên quan điểm “win - win”, có nghĩa nếu để TP.HCM tắc nghẽn cả đất nước sẽ khó phát triển.\nTheo ông, để TP.HCM đứng vững trên chuỗi giá trị toàn cầu, cần tập trung vào yếu tố quyết định là con người. Trong đó, quan trọng nhất là cán bộ. “TP.HCM đang phải giải quyết vấn đề cán bộ. Khi động lực ở khu vực công tê liệt thì không thể nói hệ thống kinh tế tư nhân có thể hoạt động, bởi hành chính công ngưng trệ” - ông chia sẻ và đưa ra giải pháp.\nThứ nhất, thu nhập cán bộ, công chức phải được đảm bảo. Việc tăng 80% so với mức lương trung bình vẫn còn quá thấp để thu hút người tài. Mặt khác, chỉ tăng lương là chưa đủ, phải có hệ thống đánh giá cán bộ để tiền lương tăng chảy vào đúng chỗ có năng lực nhất.\nCần xây dựng hệ thống chỉ số kết quả của từng sở, quận, phòng, xuống đến từng chuyên viên. Từ cách quản trị này sẽ có kết quả đánh giá cán bộ thực chất hơn và tạo động lực làm việc hiệu quả.\nCần phát huy tính chất đô thị của TP Thủ Đức\nTheo TS Thái Thị Tuyết Dung, cần bổ sung quy định chi tiết về cơ chế đặc thù cho TP Thủ Đức. Đây là đơn vị TP thuộc TP trực thuộc trung ương duy nhất hiện nay, gánh vác không chỉ trọng trách đô thị vệ tinh của vùng đô thị TP.HCM, cực tăng trưởng mạnh mẽ thúc đẩy vai trò đầu tàu kinh tế, mà còn gánh trách nhiệm chứng minh tính hiệu quả của mô hình đơn vị hành chính TP thuộc TP trên cả nước.\nQua hơn 1,5 năm hình thành, chính quyền TP Thủ Đức vẫn là cấp huyện, chưa có sự đột phá. Thậm chí các thủ tục hành chính tại TP Thủ Đức kéo dài thời gian hơn so với trước khi sáp nhập; chưa có sự phân cấp, phân quyền, ủy quyền; chưa có cơ chế đột phá phát huy tính chất đô thị.\nĐể TP thuộc TP trở thành một cú hích pháp lý và phát triển đúng với nhiệm vụ, mục tiêu, kỳ vọng, cần thí điểm những thay đổi lớn dựa trên lý thuyết về tổ chức chính quyền đô thị.\n“Có thể thí điểm cơ chế bầu trực tiếp người đứng đầu TP thuộc TP; HĐND TP thuộc TP được quyền quyết định những nội dung thuộc thẩm quyền của HĐND cấp tỉnh” - bà gợi mở."
    ]
    # texts = [
    # "Ủy ban Giám sát kế toán các công ty đại chúng Mỹ (PCAOB) ngày 26/8 thông báo cơ quan này đã ký thỏa thuận với các nhà quản lý Trung Quốc để cho phép các cơ quan quản lý Mỹ kiểm tra và điều tra các công ty kế toán đã đăng ký ở Trung Quốc và Khu hành chính đặc biệt Hong Kong (Trung Quốc). Hơn 10 năm qua, giới chức Mỹ đã yêu cầu được tiếp cận các tài liệu kiểm toán của các công ty Trung Quốc niêm yết tại Mỹ, nhưng Trung Quốc không muốn để các cơ quan quản lý nước ngoài thanh tra các công ty kế toàn của nước này do lo ngại về an ninh quốc gia.\nThỏa thuận trên đánh dấu sự \"tan băng\" một phần trong mối quan hệ giữa hai nước, và cũng là một sự xoa dịu đối với các công ty Trung Quốc, giới đầu tư và các sàn giao dịch của Mỹ, khi nó cho Trung Quốc cơ hội được tiếp tục tiếp cận thị trường vốn lớn nhất thế giới nếu thỏa thuận trên phát huy hiệu quả trong thực tiễn. Nếu không, Chủ tịch Ủy ban Giao dịch và Chứng khoán Mỹ (SEC) Gary Gensler cho biết khoảng 200 công ty của Trung Quốc có thể bị cấm khỏi các sàn giao dịch của Mỹ. Trước đó, SEC đã đưa tập đoàn Alibaba Group, JD.Com Inc, và NIO INC vào danh sách có công ty có nguy cơ này.\nGiới chức Mỹ vẫn tỏ ra thận trọng khi công bố thỏa thuận trên, cảnh báo đây chỉ là bước đi đầu tiên và sự đánh giá của phía Mỹ về sự thuân thủ thỏa thuận của Trung Quốc sẽ dựa trên việc liệu các cơ quan quản lý Mỹ có được thực hiện việc kiểm tra mà không gặp trở ngại như thỏa thuận đã cam kết hay không.\nThế nhưng, theo PCAOB, đây vẫn là thỏa thuận chi tiết và mang tính quy tắc nhất mà PCAOB từng đạt được với Trung Quốc. Về phía mình, Ủy ban quản lý chứng khoán Trung Quốc (CSRC) cũng cho biết thỏa thuận trên là một bước quan trọng hướng đến việc giải quyết vấn đề về kiểm toán và đem lại lợi ích cho các nhà đầu tư, các doanh nghiệp và cả hai nước.\nVề mặt nguyên tắc, thỏa thuận trên đáp ứng yêu cầu lâu nay của PCAOB là được tiếp cận đầy đủ các tài liệu kiểm toán không bị chỉnh sửa của Trung Quốc, được lấy lời khai từ nhân viên của các công ty kiểm toán ở nước này và có quyền tùy ý lựa chọn công ty nào bị kiểm tra.\nGiới chức Mỹ cho biết đã thông báo các công ty được chọn vào ngày 26/8 và dự kiến sẽ đến Hong Kong, nơi thực hiện các cuộc kiểm tra, vào giữa tháng Chín.",
    # "Bắc Kinh và Washington đã đạt được thỏa thuận sơ bộ cho phép giới chức Mỹ thanh tra tài liệu kiểm toán của các doanh nghiệp Trung Quốc đang niêm yết cổ phiếu tại New York. Thỏa thuận là bước đầu tiên, giúp gần 200 công ty Trung Quốc, bao gồm những tập đoàn lớn như Alibaba, Baidu, có tổng vốn hóa gần 1.000 tỉ đô la, tránh được nguy cơ hủy niêm yết bắt buộc khỏi sàn giao dịch chứng khoán New York. Thỏa thuận được ký kết giữa Ủy ban Quản lý chứng khoán Trung Quốc (CSRC), Bộ Tài chính Trung Quốc và Ủy ban Giám sát kế toán công ty đại chúng Mỹ (PCAOB). Thỏa thuận không cho phép Trung Quốc giữ lại hoặc chỉnh sửa bất kỳ thông tin nào có trong tài liệu kiểm toán vì bất kỳ lý do gì, đồng thời cho phép PCAOB phỏng vấn trực tiếp nhân viên các công ty kiểm toán ở Trung Quốc và Hong Kong để thanh tra công việc kiểm toán. PCAOB cũng có thể chuyển thông tin kiểm toán của các công ty Trung Quốc đến Ủy ban Chứng khoán và các sàn giao dịch Mỹ (SEC).\nMặc dù Trung Quốc không được phép biên tập thông tin kiểm toán, nhưng CSRC có thể xếp loại một số dữ liệu, bao gồm thông tin cá nhân, là “bị hạn chế” và chỉ một số ít thanh tra viên của Mỹ đươc phép “xem” chúng, chứ không được sao chép. Các quan chức PCAOB cho biết ủy ban này vẫn có thể thu thập các dữ liệu bị hạn chế khi cần thiết nhưng phải theo một thủ tục đặc biệt .\nPCAOB có toàn quyền lựa chọn các công ty Trung Quốc có cổ phiếu niêm yết tại Mỹ để thanh tra. Sự lựa chọn này dựa trên đánh giá rủi ro, chẳng hạn như quy mô của công ty và lĩnh vực mà công ty hoạt động.\nThỏa thuận trên đánh dấu một bước đột phá lớn trong sự bế tắc kéo dài hàng thập niên giữa hai nền kinh tế lớn nhất thế giới về quyền tiếp cận tài liệu kiểm toán. Cuộc tranh cãi kéo dài vấn đề kiểm toán đã trở thành điểm nghẽn chính trị sau khi đạo luật về trách nhiệm giải trình các công ty nước ngoài được Mỹ thông qua vào năm 2020, cho phép SEC hủy niêm yêt cổ phiếu đối với những công ty nước ngoài ngăn cản giới chức Mỹ thanh tra tài liệu kiểm toán của họ trong 3 năm liên tục.\nTrên thực tế, đạo luật này chủ yếu nhắm đến các công ty Trung Quốc đang có cổ phiếu giao dịch ở Phố Wall vì Trung Quốc và Hong Kong là hai khu vực pháp lý duy nhất trên toàn thế giới không cho phép PCAOB kiểm tra tài liệu kiểm toán của họ do lo ngại thông tin về an ninh và bí mật quốc gia bị lộ.\nThỏa thuận trên thể hiện một sự thỏa hiệp hiếm hoi về vấn đề này từ Bắc Kinh.\n“Chúng tôi đã ký kết thỏa thuận chưa từng có tiền lệ vào sáng nay, nhưng chúng tôi vẫn cần xem trong vài tháng tới liệu Trung Quốc có tuân thủ hay không”, Chủ tịch SEC, Gary Gensler cho biết trong một cuộc phỏng vấn với Bloomberg hôm 26-8. Ông nói thỏa thuận này toàn diện hơn so với bất kỳ nước nào khác và đến tháng 12 tới, các quan chức Mỹ sẽ đánh giá liệu họ có được tiếp cận đầy đủ tài liệu kiểm toán của các công ty Trung Quốc hay không.\nChủ tịch PCAOB, Erica Williams nói: “Không có điều chỉnh đặc biệt nào với Trung Quốc. Chúng tôi không cung cấp cho Trung Quốc bất cứ thứ gì mà chúng tôi không cung cấp cho các nước khác trên thế giới”.\nCác quan chức Mỹ nhấn mạnh rằng thỏa thuận mới chỉ là bước đầu tiên. PCAOB sẽ cần phải có một số lượng lớn thanh tra viên và việc đánh giá đánh giá các công ty Trung Quốc được chọn lọc có thể mất hàng tháng.\nCùng ngày, CSRC ra tuyên bố nhấn mạnh các công ty Trung Quốc có thể tránh được nguy cơ hủy niêm yết bắt buộc ở Mỹ nếu sự hợp tác sau đó có thể làm hài lòng cả đôi bên. Tuyên bố các bên đã thiết lập các quy tắc rõ ràng để xử lý thông tin nhạy cảm trong tài liệu kiểm toán của các công ty Trung Quốc.\nTrong phiên giao dịch ngày hôm qua, giá cổ phiếu của những “gã khổng lồ” công nghệ và thương mại điện tử của Trung Quốc gồm Alibaba, JD.com và Pinduoduo của Trung Quốc ban đầu tăng mạnh từ 4-6%. Nhưng sau đó, thành quả tăng điểm này bị xóa sạch do thị trường chứng khoán Mỹ lao dốc trước phát biểu bày tỏ quan điểm tiếp tục tăng mạnh lãi suất để kiềm chế lạm phát của Chủ tịch Cục Dự trữ liên bang Mỹ (Fed) Jerome Powell.\nCác cuộc đàm phán giữa Bắc Kinh và Washington về vấn đề kiểm toán tăng tốc sau khi hôm 12-8, 5 tập đoàn nhà nước Trung Quốc thông báo kế hoạch hủy niêm yết tại Mỹ.\nPCAOB tiết lộ các thanh tra viên đang chuẩn lên đường đến Hong Kong để bắt đầu thanh tra tài liệu kiểm toán. PCAOB giải thích thành phố được chọn vì quy trình kiểm dịch liên quan đến Covid-19 dễ dàng hơn so với Trung Quốc.\nPCAOB, một tổ chức phi lợi nhuận, có trụ sở ở Washsington, được thành lập cách đây 20 năm theo đạo luật Sarbanes-Oxley 2002, vốn được thông qua sau vụ bê bối kiểm toán của Công ty năng lượng Enron, có trụ sở tại bang Texas. Đạo luật nhằm ngăn ngừa các gian lận sổ sách tài chính của các công ty đại chúng có thể gây mất mát lớn cho cổ đông.\nTheo Reuters, Bloomberg\nChánh Tài",
    # "Gần 200 công ty Trung Quốc có cổ phiếu giao dịch tại Mỹ nguy cơ đối mặt với việc hủy niêm yết khi Trung Quốc tìm cách tránh sự giám sát tài chính của các cơ quan quản lý Mỹ. Theo hãng thông tấn Anadolu (Thổ Nhĩ Kỳ) ngày 26/8, khi các công ty quốc doanh lớn của Trung Quốc gần đây tuyên bố sẽ nộp đơn xin hủy niêm yết trên thị trường chứng khoán Mỹ, sự không chắc chắn đã xuất hiện về việc liệu đây có phải là dấu hiệu cho thấy sự tách biệt đang tăng gia giữa hai nền kinh tế lớn nhất thế giới.\nCác quyết định thông báo vào tuần trước từ năm công ty lớn, bao gồm cả các công ty dầu mỏ khổng lồ của Trung Quốc, được đưa ra trong bối cảnh không hài lòng về một luật của Mỹ cho phép các cơ quan quản lý Mỹ kiểm tra các cuộc kiểm toán của những công ty Trung Quốc.\nCác công ty như PetroChina, China Life Insurance, Aluminium Corporation của Trung Quốc, Sinopec và công ty con của Sinopec là Shanghai Petrochemical Co. cho biết họ đang có kế hoạch rời khỏi sàn giao dịch chứng khoán New York trong tháng này.\nNhiều công ty Trung Quốc khác có thể làm theo vì Trung Quốc đã từ chối cho phép các cơ quan quản lý ở nước ngoài kiểm tra kế toán, viện dẫn luật an ninh của nhà nước để ngăn chặn các cơ quan quản lý của Mỹ tiến hành các cuộc thanh tra này.\nAndrew KP Leung, một chiến lược gia độc lập về Trung Quốc tại Hồng Kông, cho biết: “Đây là sự khởi đầu của làn sóng mới về việc một số công ty lớn hủy niêm yết khỏi Mỹ sau sự quấy rối và cưỡng bức các doanh nghiệp khác nhau của Washington”.\nChiến lược gia quốc tế trên nói thêm rằng việc Washington viện cớ rằng các công ty bị cáo buộc tuân thủ luật pháp của Mỹ đã \"khiến rất nhiều công ty Trung Quốc sợ hãi\".\nCác kiểm toán viên của những công ty giao dịch công khai ở Mỹ được kiểm soát kỹ bởi các cơ quan quản lý của Mỹ theo Đạo luật chịu trách nhiệm về các công ty nước ngoài (HFCAA) được thông qua vào năm 2020.\nLuật cấm giao dịch chứng khoán của các công ty không đáp ứng các yêu cầu kiểm toán trong ba năm liên tiếp, với năm 2024 được đặt là thời hạn cuối cùng cho các công ty Trung Quốc cần phải lựa chọn giữa tuân thủ hoặc hủy niêm yết.\nDo đó, khoảng 200 công ty Trung Quốc có thể phải hủy niêm yết nếu các bên không thể thỏa hiệp sớm hoặc đây sẽ là một quá trình mà Trung Quốc sẽ phải lựa chọn công ty nào mà họ sẽ cho phép niêm yết tại Mỹ.\nTheo Leung, HFCAA sẽ dẫn đến việc \"rút lui khỏi thị trường chứng khoán Mỹ\", nhưng nó có thể \"không ảnh hưởng đến nhập khẩu từ Trung Quốc vào Mỹ, vì một số lệnh cấm nhập khẩu của chính quyền Joe Biden đã gây tổn hại cho người tiêu dùng Mỹ, thúc đẩy lạm phát\".\nÔng ông Leung cho rằng điều này không đủ để \"phủ bóng đen\" lên các công ty Trung Quốc và “ngày càng nhiều quốc gia không muốn nghe những lời hùng biện của Mỹ\". Nhưng việc tách rời sẽ không dễ dàng.\nÔng Leung lưu ý, mọi người đang phân biệt \"giữa lời nói khoa trương và kinh doanh thực tế\", chỉ ra sự gia tăng xuất khẩu của Trung Quốc trong một năm tài chính rưỡi vừa qua sang cả Mỹ và các nước khác. Tuy nhiên, ông thừa nhận rằng EU có thể \"thắt chặt một số quy tắc\".\nVề phần mình, Einar Tangen, một thành viên cấp cao từ Viện Taihe có trụ sở tại Bắc Kinh, nhận định rằng Mỹ đang tăng cường chương trình \"Nước Mỹ trên hết\" (America First).\nChuyên gia này nhấn mạnh: \"Một khía cạnh mà Mỹ và Trung Quốc chưa đạt được nhiều tiến bộ là tiêu chuẩn kế toán có thể chấp nhận được đối với các công ty Trung Quốc. Thứ hai, là mức độ chung của áp lực kinh tế, chính trị và an ninh mà Bắc Kinh nhận thấy đến từ Mỹ. Điều này khiến các công ty tuyên bố hủy niêm yết của Trung Quốc đã làm như vậy như là cách để phòng thủ\"."
    # ]
    golden = "Nghị quyết 54 sắp hết hiệu lực vào cuối năm 2022 nhưng hiện vẫn chưa có dự thảo Nghị quyết mới thay thế, dẫn đến tiềm ẩn nguy cơ trần nợ công sẽ bị hạ xuống đối với TP.HCM. TP. HCM hết room tín dụng, vướng mắc trong khâu huy động vốn vay, hiện mới giải ngân được 6% của gói 347.000 tỷ đồng hỗ trợ chương trình phục hồi và phát triển kinh tế - xã hội được Chính phủ trình Quốc hội nhằm khôi phục nhanh chuỗi sản xuất, tạo sức bật tăng trưởng. Nhiều cơ chế, chính sách của NQ 54 chưa được tận dụng triệt để dẫn tới hiệu quả chưa cao, hơn 4 năm qua thành phố không được hưởng một đồng nào từ khoản thu tiền sử dụng đất khi bán các tài sản công của các đơn vị trung ương trên địa bàn, việc cổ phần hoá các doanh nghiệp nhà nước cũng không thực hiện được do chưa có hướng dẫn cụ thể từ trung ương. Các chuyên gia đã có nhiều góp ý cho nghị quyết mới thay thế Nghị quyết 54. Thay vì xin cơ chế giải quyết những vấn đề riêng lẻ, trao quyền nhỏ giọt, cần kiến nghị cho phép HĐND TP. HCM được quyền quyết định các vấn đề quan trọng."
    # golden = """Hơn 10 năm qua, giới chức Mỹ đã yêu cầu được tiếp cận các tài liệu kiểm toán của các công ty Trung Quốc niêm yết tại Mỹ nhưng Trung Quốc không muốn do lo ngại về an ninh quốc gia. Thoả thuận được ký kết giữa Uỷ ban Quản lý chứng khoán Trung Quốc (CSRC), Bộ Tài chính Trung Quốc và Uỷ ban Giám sát kế toán công ty đại chúng Mỹ (PCAOB) đã đánh dấu sự " tan băng " một phần trong mối quan hệ giữa hai nước. Thoả thuận không cho phép Trung Quốc giữ lại hoặc chỉnh sửa bất kỳ thông tin nào có trong tài liệu kiểm toán vì bất kỳ lý do gì, đồng thời cho phép PCAOB phỏng vấn trực tiếp nhân viên các công ty kiểm toán ở Trung Quốc và Hong Kong để thanh tra công việc kiểm toán cũng như chuyển thông tin kiểm toán của các công ty Trung Quốc đến Uỷ ban Chứng khoán và các sàn giao dịch Mỹ (SEC). Trong khi đó, các công ty quốc doanh lớn của Trung Quốc gần đây tuyên bố sẽ nộp đơn xin huỷ niêm yết trên thị trường chứng khoán Mỹ, tạo ra lo ngại sự tách biệt đang tăng gia giữa hai nền kinh tế lớn nhất thế giới. Luật cấm giao dịch chứng khoán của các công ty không đáp ứng các yêu cầu kiểm toán trong ba năm liên tiếp, với năm 2024 được đặt là thời hạn cuối cùng cho các công ty Trung Quốc cần phải lựa chọn giữa tuân thủ hoặc huỷ niêm yết."""
    result = MultiDocSummarizationAPI(texts, 0.15, golden)
    print(result)

