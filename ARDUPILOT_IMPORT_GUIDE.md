# ArduPilot Video Import Guide

## Generated Video Files
The animation script now creates multiple video formats for maximum compatibility:
- `circumnavigation_animation.mp4` - H.264 encoded MP4 (recommended)
- `circumnavigation_animation.mov` - QuickTime MOV format
- `circumnavigation_animation.avi` - AVI format for older systems

## Importing into ArduPilot Ground Control Software

### 1. Mission Planner (Windows/Linux via Wine)
- Open Mission Planner
- Go to **Flight Data** tab
- Right-click on the map area
- Select **Set Background Image** or **Video Overlay**
- Browse and select your video file (use .mp4 for best compatibility)
- Adjust transparency and positioning as needed

### 2. QGroundControl (Recommended for Ubuntu)
- Install QGroundControl: `sudo snap install qgroundcontrol-herelink`
- Open QGroundControl
- Go to **Application Settings**
- Select **Video** tab
- Set video source to **File**
- Browse and select your .mp4 file
- Configure playback settings

### 3. MAVProxy (Command Line)
```bash
# Install MAVProxy if not already installed
pip install MAVProxy

# Start MAVProxy with video overlay
mavproxy.py --console --map --load-module video --video-file circumnavigation_animation.mp4
```

### 4. Using with ArduPilot SITL (Software In The Loop)
```bash
# Navigate to your ArduPilot directory
cd ~/ardupilot

# Start SITL with video overlay capability
sim_vehicle.py -v ArduCopter --console --map --add-param-file=your_params.parm

# In MAVProxy console, load video module
module load video
video set circumnavigation_animation.mp4
```

## Video Specifications
- **Format**: H.264 MP4
- **Frame Rate**: 10 FPS
- **Pixel Format**: YUV420P (widely compatible)
- **Codec**: libx264 with CRF 23 (good quality/size balance)

## Troubleshooting

### If video doesn't play:
1. **Check codec support**: Install additional codecs
   ```bash
   sudo apt install ubuntu-restricted-extras
   sudo apt install ffmpeg
   ```

2. **Convert to different format**:
   ```bash
   ffmpeg -i circumnavigation_animation.mp4 -c:v libx264 -preset slow -crf 22 -c:a aac output.mp4
   ```

3. **For older systems**, use the AVI file instead

### Performance Issues:
- Use the MP4 file for best performance
- Reduce video quality if needed:
  ```bash
  ffmpeg -i circumnavigation_animation.mp4 -vf scale=640:480 -c:v libx264 -crf 28 smaller_video.mp4
  ```

## Alternative: Using as Background in Mission Planning
1. Export individual frames and use as map tiles
2. Convert to image sequence for frame-by-frame analysis
3. Overlay GPS coordinates for georeferenced video

## Log File Integration
To correlate with actual flight logs:
1. Export telemetry data from your ArduPilot logs
2. Sync timestamps between video and log data
3. Use tools like MAVExplorer for synchronized playback
