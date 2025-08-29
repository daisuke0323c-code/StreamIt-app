from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Excel → Markdown 変換", layout="wide")
st.title("Excel → Markdown 変換")

uploaded = st.file_uploader(
    "Excel ファイルをアップロード",
    type=["xlsx", "xls", "xlsm"],  # ← xlsm を追加
    accept_multiple_files=False
)

def read_excel_any(file) -> pd.ExcelFile:
    suffix = Path(file.name).suffix.lower()
    if suffix in [".xlsx", ".xlsm"]:
        # xlsx / xlsm は openpyxl
        return pd.ExcelFile(file, engine="openpyxl")
    elif suffix == ".xls":
        # 旧 xls は xlrd
        return pd.ExcelFile(file, engine="xlrd")
    else:
        raise ValueError(f"未対応の拡張子: {suffix}")

if uploaded is not None:
    try:
        xls = read_excel_any(uploaded)
        st.success(f"読み込み成功：{uploaded.name}（シート数: {len(xls.sheet_names)}）")

        for sheet in xls.sheet_names:
            st.subheader(f"📄 シート: {sheet}")
            # 1行目がヘッダか不明な場合は header=None にしておく
            df = xls.parse(sheet, header=None)
            # NaN を空文字にして見やすく
            md = df.fillna("").to_markdown(index=False)

            st.code(md, language="markdown")
            # ダウンロードも用意
            fname = f"{Path(uploaded.name).stem}_{sheet}.md"
            st.download_button(
                label=f"↓ {sheet} をMarkdownで保存",
                data=md.encode("utf-8"),
                file_name=fname,
                mime="text/markdown"
            )

    except Exception as e:
        st.error(f"読み込みに失敗しました：{e}")
        st.info("ヒント：.xlsm の場合は openpyxl が必要です。xls の場合は xlrd が必要です。")
