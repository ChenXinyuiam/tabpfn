# GitHub Upload Checklist

Use this checklist before publishing the project to GitHub.

## 1. Review local changes

```bash
git status --short
git diff -- README.md .gitignore docs/GITHUB_UPLOAD.md
```

## 2. Stage intended source and docs

```bash
git add README.md .gitignore docs/GITHUB_UPLOAD.md pyproject.toml uv.lock main.py src utils templates
git add -u
```

`git add -u` records intentional deletions and modifications of files that were already tracked. Run `git status --short` again before committing.

## 3. Commit

```bash
git commit -m "Prepare project for GitHub upload"
```

## 4. Create an empty GitHub repository

Create a new empty repository on GitHub, for example:

```text
https://github.com/ChenXinyuiam/tabpfn.git
```

Do not initialize it with a README, license, or `.gitignore` if this local repository already contains those files.

## 5. Add remote and push

```bash
git remote add origin https://github.com/ChenXinyuiam/tabpfn.git
git branch -M main
git push -u origin main
```

If `origin` already exists, replace the URL:

```bash
git remote set-url origin https://github.com/ChenXinyuiam/tabpfn.git
```

## Notes

- Raw data, generated data, model checkpoints, trained `.pkl` files, and experiment outputs are ignored by default.
- GitHub blocks files larger than 100 MB. Keep large artifacts outside Git or publish them through Releases/cloud storage.
- Add a `LICENSE` file before making the repository public if reuse by others is intended.
