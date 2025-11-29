# FootAI

**FootAI** is an advanced Machine Learning project designed for precise tracking of players and the ball during football matches. It combines state-of-the-art computer vision models with custom tracking and team-assignment algorithms to provide detailed analysis of football games.

---

## Key Features

### Player and Staff Detection
Utilizes a **YOLO11x** model to detect all individuals on the pitch and classify them into one of three categories:
- **Player**
- **Goalkeeper**
- **Referee**

### Ball Tracking
The **RetinaNet** model is employed to reliably detect and track the ball in each frame, even in fast-paced sequences.

### Pitch Keypoint Detection
The **YOLO11x-pose** model identifies critical pitch landmarks, including:
- Corners
- Halfway line
- Center circle
- Penalty areas
- Penalty marks
- Goal areas

### Team Assignment
Players are automatically assigned to their respective teams using a vector-based algorithm. The method works by:
1. Creating an embedding vector for each team.
2. Comparing cropped images of detected players against these vectors.
3. Assigning the player to the team with the closest match.

---

## Technical Highlights

- Combines **YOLO** and **RetinaNet** models for robust multi-object detection.
- Custom **tracking module** ensures consistent player IDs across frames.
- Embedding-based **team classification** for accurate assignment even in complex formations.
- Pitch keypoint detection enables generation of **heatmaps, tactical analysis, and positional statistics**.

---

## Use Cases

- Player performance analysis
- Tactical breakdowns
- Match event visualization
- Automated highlight generation
- Sports analytics research

---

## System Workflow

```mermaid
flowchart TD
    A[Video Input] --> B[YOLO11x Detection]
    B --> C[Classify: Player / Goalkeeper / Referee]
    B --> D[YOLO11x-Pose: Detect Pitch Keypoints]
    A --> E[RetinaNet Ball Detection]
    C --> F[Crop Player Images]
    F --> G[Compute Embeddings]
    G --> H[Assign Players to Teams]
    D --> I[Generate Pitch Heatmaps & Stats]
    E --> I
    H --> I
