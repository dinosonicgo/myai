@echo off
rem =================================================================
rem == AI Lover 项目 - Git Tag 创建与推送脚本 v4.0               ==
rem =================================================================
rem == 功能:                                                     ==
rem == 1. 将当前代码页切换到 UTF-8。                             ==
rem == 2. 检查本地是否与远程完全同步 (這是關鍵檢查!)。         ==
rem == 3. 提示用户输入 Tag 的名称 (版本号) 和详细中文说明。      ==
rem == 4. 为当前最新的 commit 创建一个附注 Tag。                 ==
rem == 5. 将所有本地 Tags 推送到远程 GitHub 仓库。               ==
rem =================================================================

rem --- 步骤 1: 设置环境以支持中文 ---
chcp 65001 > nul
echo.
echo =================================================
echo      AI Lover Git Tag (版本快照) 创建工具
echo =================================================
echo.

rem --- 步骤 2: 检查同步状态 ---
echo [步骤 1/4] 正在检查并同步远程仓库状态...
git fetch origin
git status -uno | findstr /C:"Your branch is up to date"
if errorlevel 1 (
    echo.
    echo [错误] 您的本地分支与远程 'origin/main' 不一致。
    echo 请先运行您的主程序，让它自动同步到最新版本后再试。
    goto :end
)
echo    ...本地已是最新版本，可以开始创建 Tag。
echo.

rem --- 步骤 3: 获取用户输入的 Tag 信息 ---
echo [步骤 2/4] 请为当前的稳定版本快照命名。
set /p tag_name="Tag 名称 (例如: v1.8, v2.0-stable): "
if not defined tag_name (
    echo.
    echo [错误] Tag 名称不能为空! 操作已取消。
    goto :end
)
echo.

echo [步骤 3/4] 请为这个 Tag 添加详细的中文说明。
set /p tag_message="Tag 说明 (例如: '完成了速率限制的修复，此版本稳定'): "
if not defined tag_message (
    echo.
    echo [错误] Tag 说明不能为空! 操作已取消。
    goto :end
)
echo.


rem --- 步骤 4: 创建附注 Tag ---
echo [步骤 4/4] 正在创建 Tag '%tag_name%'...
git tag -a "%tag_name%" -m "%tag_message%"
if errorlevel 1 (
    echo.
    echo [错误] 'git tag' 创建失败。这通常意味着该 Tag 名称 '%tag_name%' 已经存在。
    goto :end
)
echo    ...Tag '%tag_name%' 创建成功!
echo.

rem --- 步骤 5: 推送到 GitHub ---
echo [步骤 5/5] 正在将所有本地 Tags 推送到 GitHub...
git push origin --tags
if errorlevel 1 (
    echo.
    echo [错误] 'git push --tags' 失败。请检查您的网络连接和 GitHub 权限。
    goto :end
)
echo    ...推送成功!
echo.

rem --- 结束 ---
echo =================================================
echo      Tag '%tag_name%' 已成功发布到 GitHub!
echo =================================================
echo.

:end
pause
