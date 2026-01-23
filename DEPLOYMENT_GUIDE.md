# 🚀 Deploy Alzheimer's Detection App to Render

This guide will walk you through deploying your Flask-based Alzheimer's Disease Detection application to Render.

## 📋 Prerequisites

- ✅ GitHub account
- ✅ Git installed on your computer
- ✅ Render account (free - sign up at https://render.com)

---

## 🎯 Step-by-Step Deployment Guide

### **Step 1: Push Your Code to GitHub**

First, ensure your code is in a GitHub repository.

```bash
# Navigate to your project directory
cd "c:\Users\RAJAGANAPATHY\Alzheimer Ai\CNN_design_for_AD"

# Initialize git if not already done
git init

# Add all files
git add .

# Commit your changes
git commit -m "Prepare for Render deployment"

# Add your GitHub repository as remote (replace with your actual repo URL)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push to GitHub
git push -u origin main
```

> **Note:** If you already have a GitHub repository, just ensure your latest changes are pushed.

---

### **Step 2: Sign Up for Render**

1. Go to [https://render.com](https://render.com)
2. Click **"Get Started"**
3. Sign up using your **GitHub account** (recommended for easy integration)

---

### **Step 3: Create a New Web Service**

1. After logging in, click **"New +"** in the top right
2. Select **"Web Service"**
3. Connect your GitHub repository:
   - Click **"Connect repository"**
   - Select your Alzheimer's Detection repository
   - Click **"Connect"**

---

### **Step 4: Configure Your Web Service**

Render will auto-detect your `render.yaml` file. Verify these settings:

| Field | Value |
|-------|-------|
| **Name** | `alzheimer-detection-app` (or your choice) |
| **Runtime** | `Python 3` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `gunicorn --bind 0.0.0.0:$PORT --timeout 120 --workers 1 web_app.app:app` |
| **Plan** | `Free` |

---

### **Step 5: Add Environment Variables (Optional)**

If needed, you can add environment variables:

1. Scroll to **"Environment Variables"**
2. Click **"Add Environment Variable"**
3. Add any custom variables your app needs

---

### **Step 6: Deploy!**

1. Click **"Create Web Service"**
2. Render will:
   - ✅ Clone your repository
   - ✅ Install dependencies from `requirements.txt`
   - ✅ Build your application
   - ✅ Deploy it to a live URL

This process takes **5-10 minutes** on the first deployment.

---

### **Step 7: Access Your Live App**

Once deployment completes:

1. You'll see a **green "Live"** status
2. Your app URL will be: `https://alzheimer-detection-app.onrender.com`
3. Click the URL to visit your live application! 🎉

---

## 📁 Files Created for Deployment

### `requirements.txt`
Contains all Python dependencies:
- Flask & Werkzeug (web framework)
- PyTorch (deep learning)
- nibabel (medical imaging)
- Pillow (image processing)
- gunicorn (production server)

### `render.yaml`
Render deployment configuration:
- Specifies Python runtime
- Configures build/start commands
- Sets up health checks
- Enables auto-deployment

---

## 🔄 Auto-Deployment

With `autoDeploy: true` in `render.yaml`, Render will automatically redeploy your app whenever you push to GitHub:

```bash
# Make changes to your code
git add .
git commit -m "Update feature"
git push

# Render automatically detects the push and redeploys! ✨
```

---

## ⚠️ Important Notes

### **Model Files**
- Your trained model files in `saved_model/` need to be included in your Git repository
- If your models are too large (>100MB), you may need to use Git LFS:
  ```bash
  git lfs install
  git lfs track "*.pth.tar"
  git add .gitattributes
  ```

### **Free Tier Limitations**
Render's free tier:
- ✅ 750 hours/month of runtime
- ✅ Sleeps after 15 minutes of inactivity
- ⏱️ Cold start: ~30 seconds when app wakes up
- 💾 512 MB RAM

---

## 🐛 Troubleshooting

### Build Fails
**Check the build logs** in Render dashboard:
- Look for missing dependencies
- Verify Python version compatibility

### App Won't Start
- Ensure `gunicorn` is in `requirements.txt` ✅ (already added)
- Check start command syntax in `render.yaml`

### Health Check Fails
- Your Flask app must respond to HTTP requests on `0.0.0.0:$PORT`
- We've updated your `app.py` to handle this correctly ✅

### Model Loading Errors
- Verify `config_local.yaml` is in repository
- Check model paths in `saved_model/` directory

---

## 🎓 Alternative Deployment Options

If Render doesn't meet your needs, consider:

### **Railway** (https://railway.app)
- Similar to Render
- $5/month credit on free tier
- Great for ML apps

### **Hugging Face Spaces** (https://huggingface.co/spaces)
- Perfect for ML/AI demos
- Free GPU support available
- Great for showcasing AI projects

---

## ✅ Next Steps After Deployment

1. **Test your live app** with sample images
2. **Monitor performance** in Render dashboard
3. **Set up custom domain** (optional, available on paid plans)
4. **Enable continuous deployment** by connecting GitHub properly

---

## 📞 Need Help?

- **Render Documentation:** https://render.com/docs
- **Render Community:** https://community.render.com

---

**You're all set! 🚀** Your Alzheimer's Detection app is ready to go live on Render!
