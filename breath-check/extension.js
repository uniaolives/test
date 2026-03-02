const vscode = require('vscode');
const { exec } = require('child_process');
const path = require('path');
const fs = require('fs');

function activate(context) {
    console.log('Breath-Check is now active! Watching for breath.');

    let disposable = vscode.commands.registerCommand('breathCheck.scanFile', function () {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No file open to scan.');
            return;
        }

        const filePath = editor.document.uri.fsPath;
        const pythonScript = path.join(context.extensionPath, 'extension.py');

        vscode.window.withProgress({
            location: vscode.ProgressLocation.Window,
            title: "Scanning for breathing risks...",
            cancellable: false
        }, (progress) => {
            return new Promise((resolve) => {
                // Executa o script Python passando o caminho do arquivo
                exec(`python3 "${pythonScript}" "${filePath}"`, (error, stdout, stderr) => {
                    if (error) {
                        vscode.window.showErrorMessage(`Scanner Error: ${stderr}`);
                    } else {
                        // Exibe o resultado no canal de output
                        const outputChannel = vscode.window.createOutputChannel('Breath-Check Results');
                        outputChannel.clear();
                        outputChannel.appendLine(stdout);
                        outputChannel.show(true);

                        if (stdout.includes("CRITICAL")) {
                            vscode.window.showErrorMessage('Critical breathing risks found! Check output.');
                        } else if (stdout.includes("HIGH")) {
                            vscode.window.showWarningMessage('Potential risks found. Check output.');
                        } else {
                            vscode.window.showInformationMessage('✅ No breathing risks detected.');
                        }
                    }
                    resolve();
                });
            });
        });
    });

    context.subscriptions.push(disposable);
}

function deactivate() {
    console.log('Breath-Check is now sleeping.');
}

module.exports = {
    activate,
    deactivate
};
