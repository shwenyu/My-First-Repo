import numpy as np

def needleman_wunsch(seq1, seq2, match_score=1, gap_penalty=-1, mismatch_penalty=-1):
    # 创建评分矩阵
    n = len(seq1) + 1
    m = len(seq2) + 1
    score_matrix = np.zeros((n, m), dtype=int)
    
    # 初始化边界条件（gap penalties）
    for i in range(n):
        score_matrix[i][0] = i * gap_penalty
    for j in range(m):
        score_matrix[0][j] = j * gap_penalty
    
    # 填充评分矩阵
    for i in range(1, n):
        for j in range(1, m):
            match = score_matrix[i-1][j-1] + (match_score if seq1[i-1] == seq2[j-1] else mismatch_penalty)
            delete = score_matrix[i-1][j] + gap_penalty
            insert = score_matrix[i][j-1] + gap_penalty
            score_matrix[i][j] = max(match, delete, insert)
    
    # 回溯找最优比对路径
    aligned_seq1 = []
    aligned_seq2 = []
    i, j = n - 1, m - 1
    while i > 0 and j > 0:
        current_score = score_matrix[i][j]
        if current_score == score_matrix[i-1][j-1] + (match_score if seq1[i-1] == seq2[j-1] else mismatch_penalty):
            aligned_seq1.append(seq1[i-1])
            aligned_seq2.append(seq2[j-1])
            i -= 1
            j -= 1
        elif current_score == score_matrix[i-1][j] + gap_penalty:
            aligned_seq1.append(seq1[i-1])
            aligned_seq2.append('-')
            i -= 1
        else:
            aligned_seq1.append('-')
            aligned_seq2.append(seq2[j-1])
            j -= 1
    
    # 处理剩余部分
    while i > 0:
        aligned_seq1.append(seq1[i-1])
        aligned_seq2.append('-')
        i -= 1
    while j > 0:
        aligned_seq1.append('-')
        aligned_seq2.append(seq2[j-1])
        j -= 1
    
    # 返回比对结果
    return ''.join(reversed(aligned_seq1)), ''.join(reversed(aligned_seq2)), score_matrix[n-1][m-1]

# 示例
seq1 = "GATTACA"
seq2 = "GCATGCU"
aligned_seq1, aligned_seq2, score = needleman_wunsch(seq1, seq2)

print("Aligned Sequence 1:", aligned_seq1)
print("Aligned Sequence 2:", aligned_seq2)
print("Alignment Score:", score)