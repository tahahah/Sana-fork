# Running Pacman Inference over SSH

This guide explains how to run the Pacman inference script over SSH connections.

## Options

The script now supports several options for running over SSH:

### 1. X11 Forwarding (Graphical Display)

This option allows you to see the graphical display over SSH.

```bash
# Connect to the remote server with X11 forwarding
ssh -X username@remote_server

# Run the script with X11 display enabled
python inference_pacman.py --x11_display
```

Requirements:
- SSH with X11 forwarding enabled (`ssh -X` or `ssh -Y`)
- X11 server running on your local machine
- Working graphics drivers on the remote server

### 2. Headless Mode (Save Frames)

This option runs without a display and saves all frames to disk.

```bash
# Run in headless mode
python inference_pacman.py --headless --output_dir output/frames
```

You can then download the frames and create a video or view them locally.

### 3. Save Frames While Displaying

You can also save frames while displaying them:

```bash
python inference_pacman.py --save_frames --output_dir output/frames
```

## Creating a Video from Saved Frames

After saving frames, you can create a video using FFmpeg:

```bash
# Install FFmpeg if needed
# apt-get install ffmpeg  # Ubuntu/Debian
# yum install ffmpeg      # CentOS/RHEL

# Create video from frames
ffmpeg -framerate 10 -i output/frames/frame_%04d.png -c:v libx264 -pix_fmt yuv420p output_video.mp4
```

## Troubleshooting

### X11 Forwarding Issues

If you encounter issues with X11 forwarding:

1. Ensure X11 forwarding is enabled in your SSH config:
   ```
   # In /etc/ssh/sshd_config on the server
   X11Forwarding yes
   ```

2. Try using `ssh -Y` instead of `ssh -X` for trusted X11 forwarding

3. Check if the DISPLAY environment variable is set:
   ```bash
   echo $DISPLAY
   ```
   It should show something like `:0` or `localhost:10.0`

### Performance Issues

If the display is slow over SSH:
- Try reducing the resolution in the config file
- Use headless mode and create a video afterwards

## Command Line Arguments

```
--config           Path to the config file
--checkpoint       Path to the model checkpoint
--device           Device to run on (cuda/cpu)
--image            Initial image path
--debug            Enable debug prints
--x11_display      Enable X11 display for SSH with X forwarding
--headless         Run without display (for SSH)
--output_dir       Directory to save frames when in headless mode
--save_frames      Save frames even in non-headless mode
```
