# Script to create GitHub issues from markdown templates
# Run this after restarting terminal (so gh CLI is in PATH)

$issueFiles = @(
    ".github/issues/01-prime-number-tools.md",
    ".github/issues/02-number-theory-tools.md",
    ".github/issues/03-sequence-generator-tools.md",
    ".github/issues/04-statistical-analysis-tools.md",
    ".github/issues/05-text-processing-tools.md",
    ".github/issues/06-unit-converter-tool.md",
    ".github/issues/07-hash-generator-tool.md",
    ".github/issues/08-date-calculator-tool.md"
)

$labels = @(
    "enhancement,good first issue,math-tools",
    "enhancement,math-tools",
    "enhancement,math-tools,sequences",
    "enhancement,statistics,data-analysis",
    "enhancement,text-processing,utilities",
    "enhancement,utilities,good first issue",
    "enhancement,security,utilities",
    "enhancement,utilities,datetime"
)

for ($i = 0; $i -lt $issueFiles.Length; $i++) {
    $file = $issueFiles[$i]
    $content = Get-Content $file -Raw
    
    # Extract title (first # heading)
    $title = ($content -split "`n" | Where-Object { $_ -match "^# " } | Select-Object -First 1) -replace "^# ", ""
    
    Write-Host "Creating issue: $title"
    
    # Create issue using gh CLI
    gh issue create --title $title --body-file $file --label $labels[$i]
    
    Start-Sleep -Seconds 1
}

Write-Host "`nAll issues created successfully!"
