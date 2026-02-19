# Hugging Face Deployment Guide – Step by Step (From Scratch)

This guide walks you through deploying the Flotorch Evaluation MCP Server on Hugging Face Spaces, assuming no prior Hugging Face experience.

---

## What You’ll Need

- **Flotorch API key** – from your Flotorch account  
- **Flotorch base URL** – usually `https://gateway.flotorch.cloud`  
- **Your project code** – already pushed to Git  

---

## Part 1: Create a Hugging Face Account

1. Open **https://huggingface.co** in your browser.
2. Click **Sign up** (top right).
3. Enter your email and choose a password, or sign up with Google/GitHub.
4. Verify your email if you used email signup.
5. Log in with your new account.

---

## Part 2: Create a New Space

A **Space** is a Hugging Face app that runs your code (like a small web app).

1. Go to **https://huggingface.co/new-space**.
2. Fill in the form:

   | Field | What to enter |
   |-------|----------------|
   | **Space name** | e.g. `flotorch-eval-mcp` |
   | **License** | e.g. MIT or Apache 2.0 |
   | **SDK** | Choose **Docker** |
   | **Hardware** | **CPU basic** (free) |
   | **Visibility** | Public or Private |

3. Click **Create Space**.

4. Hugging Face will create an empty repo. You’ll see a URL like:
   ```
   https://huggingface.co/spaces/YOUR_USERNAME/flotorch-eval-mcp
   ```

---

## Part 3: Add Your Code to the Space

You need to push your project files into this Space repo.

### Before You Push – Required Files at Project Root

Your Space repo root must contain:

- `Dockerfile` (use `deploy/huggingface/Dockerfile` for port 7860, or keep root `Dockerfile` and set `PORT=7860` in Variables)
- `requirements.txt`
- `pyproject.toml`
- `src/` folder with `flotorch_eval_mcp/` inside
- `.dockerignore`

If you push your full project, the root `Dockerfile` will work as long as you add `PORT=7860` in Space Variables (Part 5).

### Option A: Push from Your Local Git (Recommended)

1. Open a terminal in your project folder:
   ```
   cd c:\Git_Files\Fission\opensource\Flotorch-Eval-Mcp-Tool
   ```

2. Add Hugging Face as a remote (replace with your username and Space name):
   ```
   git remote add huggingface https://huggingface.co/spaces/YOUR_USERNAME/flotorch-eval-mcp
   ```

3. For the first push, Hugging Face may ask for credentials. Use:
   - **Username:** your Hugging Face username  
   - **Password:** a **Token** (not your password)

4. Create a token:
   - Go to **https://huggingface.co/settings/tokens**
   - Click **New token**
   - Name it (e.g. `space-deploy`), choose **Write** access
   - Copy the token and use it as the password when Git asks

5. Push your code:
   ```
   git push huggingface main
   ```
   (Use `master` instead of `main` if your default branch is `master`.)

6. Hugging Face will build your Space automatically.

### Option B: Copy Files via Web UI

1. Open your Space URL.
2. Click **Files** (or **+ Add file**).
3. Upload or create:
   - `Dockerfile` (use `deploy/huggingface/Dockerfile` from your project)
   - `requirements.txt`
   - `pyproject.toml`
   - `.dockerignore`
   - `README.md` (see below)
   - Folder `src/` with `flotorch_eval_mcp/` and all files inside

---

## Part 4: Configure the Space README

The README must tell Hugging Face how to run your app.

1. Open **README.md** in your Space.
2. Edit it so it starts with this YAML block:

```yaml
---
title: Flotorch Evaluation MCP Server
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---
```

3. Keep the rest of your README below that block.

4. Commit and push.

---

## Part 5: Add Secrets (API Key)

Your Flotorch API key must not be in the code. Use Hugging Face secrets.

1. Open your Space URL.
2. Click **Settings** (or the gear icon).
3. Go to **Variables and secrets** in the left sidebar.
4. Add:

   | Name | Value | Type |
   |------|-------|------|
   | `PORT` | `7860` | Variable |
   | `FLOTORCH_API_KEY` | your Flotorch API key | Secret |
   | `FLOTORCH_BASE_URL` | `https://gateway.flotorch.cloud` | Variable |

5. Click **Save** for each.

---

## Part 6: Dockerfile for Hugging Face

Hugging Face Spaces expect your app to listen on port **7860**.

- **If you pushed the full project:** The root `Dockerfile` defaults to port 8080, but the server reads `PORT` from the environment. Adding `PORT=7860` as a Variable (Part 5) is enough.
- **If you want a dedicated HF Dockerfile:** Copy `deploy/huggingface/Dockerfile` over the root `Dockerfile` before pushing. It already sets `PORT=7860`.

Either approach works. No further changes needed if `PORT=7860` is set in Variables.

---

## Part 7: Wait for the Build and Check

1. After pushing, Hugging Face builds your Docker image.
2. Build status: **Building** → **Running**.
3. Wait a few minutes (first build can take 5–10 minutes).
4. When it’s running, Hugging Face will show the app URL on your Space page (e.g. **"App"** or a link). It usually looks like:
   ```
   https://YOUR_USERNAME-flotorch-eval-mcp.hf.space
   ```
   Click it to open your running app.

5. Test the discovery endpoint:
   ```
   https://YOUR_USERNAME-flotorch-eval-mcp.hf.space/.well-known/flotorch-mcp
   ```
   You should see JSON with `"transport": "HTTP_STREAMABLE"`.

---

## Part 8: Connect Your MCP Client

Use this URL and headers in your MCP client:

```
URL: https://YOUR_USERNAME-flotorch-eval-mcp.hf.space/
Headers:
  X-Flotorch-Api-Key: your_api_key
  X-Flotorch-Base-Url: https://gateway.flotorch.cloud
```

---

## Summary Checklist

- [ ] Hugging Face account created  
- [ ] New Space created with **Docker** SDK  
- [ ] Code pushed to the Space repo  
- [ ] README has YAML block with `sdk: docker` and `app_port: 7860`  
- [ ] `FLOTORCH_API_KEY` added as Secret  
- [ ] `FLOTORCH_BASE_URL` and `PORT` added as Variables  
- [ ] Build completed and Space is running  
- [ ] Discovery endpoint returns JSON  

---

## Troubleshooting

| Problem | What to do |
|---------|------------|
| Build fails | Check logs and ensure `Dockerfile`, `requirements.txt`, `pyproject.toml`, and `src/` are present. |
| "API key not found" | Add `FLOTORCH_API_KEY` as a **Secret** in Space Settings. |
| Space sleeps or stops | Free Spaces sleep after inactivity. A new request will wake it. |
| Port errors | Ensure `PORT=7860` is set and `app_port: 7860` is in the README YAML. |

---

## Need Help?

- Hugging Face docs: https://huggingface.co/docs/hub/spaces  
- Hugging Face Discord: https://huggingface.co/join/discord  
