import os
import tempfile

import pypandoc


def docx_to_markdown(input_path: str, output_path: str):
    """Word(.docx) → Markdown に変換して保存する

    Raises FileNotFoundError when input_path does not exist.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"source_file is not a valid path: {input_path}")
    # Ensure pandoc is available; try to download it automatically if missing.
    try:
        ensure_pandoc(download_if_missing=False)
    #    ensure_pandoc()
    except Exception as e:
        raise RuntimeError(
            "No pandoc was found and automatic download failed. "
            f"Install pandoc and add it to PATH or set PYPANDOC_DOWNLOAD=1. ({e})"
        ) from e

    pypandoc.convert_file(
        input_path,
        'md',
        outputfile=output_path,
        extra_args=['--standalone'],
    )
    print(f"✅ {input_path} → {output_path} に変換完了")


def ensure_pandoc(download_if_missing: bool = True):
    """Check pandoc availability and optionally download it.

    Raises RuntimeError on failure.
    """
    try:
        ver = pypandoc.get_pandoc_version()
        # If version returned, pandoc is available.
        return ver
    except OSError:
        if not download_if_missing:
            raise RuntimeError("pandoc not found")

        # Try automatic download (pypandoc will place it in user cache)
        try:
            pypandoc.download_pandoc()
            # verify
            return pypandoc.get_pandoc_version()
        except Exception as e:
            raise RuntimeError(f"pypandoc.download_pandoc() failed: {e}") from e


# Streamlit 対応（アップロード→一時ファイルに書き出して pypandoc に渡す）
try:
    import streamlit as st  # type: ignore
    _HAS_STREAMLIT = True
except Exception:
    _HAS_STREAMLIT = False


if _HAS_STREAMLIT:
    st.title("Word(.docx) → Markdown 変換")
    uploaded = st.file_uploader("Word ファイルをアップロード", type=["docx"])
    if uploaded is not None:
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        out_path = tmp_path + ".md"
        try:
            docx_to_markdown(tmp_path, out_path)
            with open(out_path, 'rb') as f:
                md_bytes = f.read()
            st.download_button("Markdown をダウンロード", data=md_bytes, file_name=os.path.basename(out_path), mime="text/markdown")
        except Exception as e:
            st.error(f"変換中にエラーが発生しました: {e}")
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            try:
                os.remove(out_path)
            except Exception:
                pass


else:
    if __name__ == '__main__':
        import argparse

        parser = argparse.ArgumentParser(description="Convert .docx to markdown using pypandoc")
        parser.add_argument('input', nargs='?', default=None, help='input .docx path')
        parser.add_argument('output', nargs='?', default=None, help='output .md path')
        args = parser.parse_args()

        if not args.input:
            print("Usage: python Convert_word.py input.docx [output.md]")
        else:
            input_path = args.input
            output_path = args.output or (os.path.splitext(input_path)[0] + '.md')
            try:
                docx_to_markdown(input_path, output_path)
            except FileNotFoundError as e:
                print(f"ERROR: {e}")
            except Exception as e:
                print(f"変換中にエラーが発生しました: {e}")
