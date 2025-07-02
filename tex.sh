#!/bin/bash

# LaTeX自動ビルドスクリプト
# 使用方法: ./tex.sh
# 機能: ルート直下の各フォルダ/report/report.texを降順でビルド

set -e  # エラーが発生したら即座に終了

echo "=== LaTeX自動ビルドスクリプト開始 ==="
echo "現在のディレクトリ: $(pwd)"
echo

# ルート直下のディレクトリを降順でソート
for dir in $(ls -d */ 2>/dev/null | sort -r); do
    # ディレクトリ名から末尾の"/"を削除
    dir_name=${dir%/}
    
    # report/report.texの存在チェック
    if [[ -f "${dir_name}/report/report.tex" ]]; then
        echo "📁 処理中: ${dir_name}/report/report.tex"
        
        # reportディレクトリに移動してビルド実行
        cd "${dir_name}/report"
        
        # xelatexでビルド
        if xelatex report.tex > build.log 2>&1; then
            echo "✅ ビルド成功: ${dir_name}/report/report.pdf"
            # ログファイルを削除（成功時のみ）
            rm -f build.log
        else
            echo "❌ ビルド失敗: ${dir_name}/report/report.tex"
            echo "   エラーログ: ${dir_name}/report/build.log を確認してください"
        fi
        
        # ルートディレクトリに戻る
        cd ../..
        echo
    else
        echo "⏭️  スキップ: ${dir_name} (report/report.texが存在しません)"
    fi
done

echo "=== LaTeX自動ビルドスクリプト完了 ==="