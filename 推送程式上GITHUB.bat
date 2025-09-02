@echo off
rem =================================================================
rem == AI Lover 项目 - Git 快速提交与推送脚本 v1.0              ==
rem =================================================================
rem == 功能:                                                     ==
rem == 1. 将当前代码页切换到 UTF-8 以支持中文提交信息。         ==
rem == 2. 添加所有本地更改到暂存区。                             ==
rem == 3. 提示用户输入本次快照的中文说明。                       ==
rem == 4. 创建一个新的提交 (commit)。                            ==
rem == 5. 将提交推送到远程 GitHub 仓库的 main 分支。             ==
rem =================================================================

rem --- 步骤 1: 设置环境以支持中文 ---
chcp 65001 > nul
echo.
echo =================================================
echo      AI Lover Git 快速提交与推送工具
echo =================================================
echo.
echo 当前 Git 状态:
git status -s
echo.

rem --- 步骤 2: 添加所有更改 ---
echo [步骤 1/4] 正在将所有更改添加到暂存区...
git add .
if errorlevel 1 (
    echo.
    echo [错误] git add 失败，请检查您的 Git 状态。
    goto :end
)
echo    ...完成!
echo.

rem --- 步骤 3: 获取用户输入的提交信息 ---
echo [步骤 2/4] 请输入本次快照的中文说明 (例如: "修复了 AI 角色的 LORE 感知问题"):
set /p commit_message="提交说明: "

rem 检查用户是否输入了信息
if not defined commit_message (
    echo.
    echo [错误] 提交说明不能为空! 操作已取消。
    goto :end
)
echo.

rem --- 步骤 4: 创建提交 ---
echo [步骤 3/4] 正在创建 Git 快照...
git commit -m "%commit_message%"
if errorlevel 1 (
    echo.
    echo [错误] git commit 失败。这通常意味着没有需要提交的更改。
    goto :end
)
echo    ...快照创建成功!
echo.

rem --- 步骤 5: 推送到 GitHub ---
echo [步骤 4/4] 正在将快照推送到 GitHub (origin/main)...
git push origin main
if errorlevel 1 (
    echo.
    echo [错误] git push 失败。请检查您的网络连接和 GitHub 权限。
    goto :end
)
echo    ...推送成功!
echo.

rem --- 结束 ---
echo =================================================
echo      所有操作已成功完成!
echo =================================================
echo.

:end
pause