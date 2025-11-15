import cv2
import numpy as np
import mediapipe as mp
from statistics import mean

print("working")


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

import matplotlib.pyplot as plt

def plot_angle_series(angle_list, title, filename):
   
    if not angle_list:
        print(f"No data for {title}, skipping plot.")
        return

    plt.figure(figsize=(10, 4))
    plt.plot(angle_list)
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel("Angle (degrees)")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot: {filename}")


def save_all_angle_plots(knee_angles, elbow_angles, prefix="angles"):
    """
    Saves plots for both elbow and knee angles.
    """
    plot_angle_series(knee_angles, "Knee Angle Over Time", f"{prefix}_knee.png")
    plot_angle_series(elbow_angles, "Elbow Angle Over Time", f"{prefix}_elbow.png")



#we can make more utility functions like this


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))



def process_video(video_path, exercise_type="squat", output_path="annotated_3d.mp4"):
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    # Metrics
    knee_angles, elbow_angles = [], []
    reps, stage = 0, None
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
                )
                lm = results.pose_landmarks.landmark

                # shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * width,
                #             lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * height]
                # elbow = [lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * width,
                #          lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * height]
                # wrist = [lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x * width,
                #          lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y * height]
                # hip = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x * width,
                #        lm[mp_pose.PoseLandmark.LEFT_HIP.value].y * height]
                # knee = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x * width,
                #         lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y * height]
                # ankle = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * width,
                #          lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * height]


                shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * width,
                            lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * height,
                            lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z * width]  

                elbow = [lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * width,
                         lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * height,
                         lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].z * width]      

                wrist = [lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x * width,
                         lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y * height,
                         lm[mp_pose.PoseLandmark.LEFT_WRIST.value].z * width]      

                hip = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x * width,
                       lm[mp_pose.PoseLandmark.LEFT_HIP.value].y * height,
                       lm[mp_pose.PoseLandmark.LEFT_HIP.value].z * width]          

                knee = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x * width,
                        lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y * height,
                        lm[mp_pose.PoseLandmark.LEFT_KNEE.value].z * width]       

                ankle = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * width,
                         lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * height,
                         lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].z * width]      
                # ===========================================================



                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                knee_angle = calculate_angle(hip, knee, ankle)
                elbow_angles.append(elbow_angle)
                knee_angles.append(knee_angle)

                #rep counting
                if exercise_type == "squat":
                    if knee_angle > 168:
                        stage = "up"
                    if knee_angle < 155 and stage == "up":
                        stage = "down"
                        reps += 1
                elif exercise_type == "pushup":
                    if elbow_angle > 160:
                        stage = "up"
                    if elbow_angle < 90 and stage == "up":
                        stage = "down"
                        reps += 1

                # === Draw info ===
                cv2.putText(image, f"Reps: {reps}", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
                cv2.putText(image, f"Elbow: {int(elbow_angle)} Knee: {int(knee_angle)}",
                            (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            out.write(image)
            cv2.imshow("Fitness Tracker", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    metrics = {
        "total_reps": reps,
        "avg_knee_angle": round(mean(knee_angles), 1) if knee_angles else None,
        "avg_elbow_angle": round(mean(elbow_angles), 1) if elbow_angles else None,
        "min_knee_angle": round(min(knee_angles), 1) if knee_angles else None,
        "min_elbow_angle": round(min(elbow_angles), 1) if elbow_angles else None
    }

    save_all_angle_plots(knee_angles, elbow_angles, prefix="workout")


    # rule_fb = basic_feedback(exercise_type, {"knee": knee_angles, "elbow": elbow_angles})  #for stefan to use best judgement
    # llm_fb = llm_feedback(exercise_type, metrics, rule_fb)



    print(metrics)


    # Save feedback to file
    # with open("feedback.txt", "w") as f:
    #     f.write("WORKOUT SUMMARY:\n")
    #     f.write(str(metrics) + "\n\n")
    #     f.write("FEEDBACK:\n" + llm_fb)

    print("\nAnnotated video saved to:", output_path)
    print("Feedback saved to: feedback.txt")


if __name__ == "__main__":
    
    # ============================
    # Hardcoded settings (edit here)
    # ============================
    # VIDEO_PATH = "ananya.mp4"   # <- change this
    VIDEO_PATH= "d3.mp4"
    EXERCISE_TYPE = "squat"                   # <- squat / pushup / etc.
    OUTPUT_PATH = "annotated_output.mp4"      # <- change if needed
    # ============================

    process_video(VIDEO_PATH, EXERCISE_TYPE, OUTPUT_PATH)