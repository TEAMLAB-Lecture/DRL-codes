import json
import re

def txt_to_ipynb(input_file, output_file):
    # 입력 파일 읽기
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 노트북 기본 구조 생성
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    # 텍스트를 셀로 분할
    # 마크다운 셀 시작 패턴: # %% [markdown]
    # 코드 셀 시작 패턴: # %%
    cell_patterns = re.split(r'# %%(?:\s*\[markdown\]|\s*)', content)
    cell_types = re.findall(r'# %%(?:\s*\[markdown\]|\s*)', content)
    
    # 첫 부분이 빈 문자열일 수 있으므로 제거
    if cell_patterns and not cell_patterns[0].strip():
        cell_patterns = cell_patterns[1:]
    
    for i, cell_content in enumerate(cell_patterns):
        if i >= len(cell_types):
            # 마지막 부분은 패턴이 없을 수 있음
            cell_type = "code"
        else:
            cell_type = "markdown" if "[markdown]" in cell_types[i] else "code"
        
        # 공백 제거
        cell_content = cell_content.strip()
        
        if cell_type == "markdown":
            # 마크다운 셀에서 # 기호 제거
            cell_content = re.sub(r'^#\s*', '', cell_content, flags=re.MULTILINE)
            
            # 노트북에 마크다운 셀 추가
            notebook["cells"].append({
                "cell_type": "markdown",
                "metadata": {},
                "source": cell_content.split('\n')
            })
        else:
            # 노트북에 코드 셀 추가
            notebook["cells"].append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": cell_content.split('\n')
            })

    # IPYNB 파일로 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=2)

    print(f"{input_file}을(를) {output_file}로 변환 완료!")

# 사용 예
if __name__ == "__main__":
    txt_to_ipynb("aa.txt", "데이터_전처리_예제.ipynb")