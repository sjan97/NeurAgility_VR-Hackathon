import cv2
import numpy as np
import mediapipe as mp
import os
from statistics import mean
import google.generativeai as genai
from api_key_yash import API_KEY
genai.configure(API_KEY)



mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


#we can make more utility functions like this
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

#rules for - Stefan
def basic_feedback(ex_type, angles):
    if ex_type == "squat":
        knee_angles = angles["knee"]
        avg_knee = mean(knee_angles)
        min_knee = min(knee_angles)
        if min_knee > 100:
            return f"Your squats seem shallow (min knee angle ~{int(min_knee)}°). Try going deeper."
        elif avg_knee < 80:
            return "Nice depth, keep maintaining that squat form!"
        else:
            return "Good effort! Maintain consistent squat depth."
    elif ex_type == "pushup":
        elbow_angles = angles["elbow"]
        min_elbow = min(elbow_angles)
        if min_elbow > 110:
            return f"Try lowering yourself more in pushups (min elbow angle ~{int(min_elbow)}°)."
        else:
            return "Solid pushup depth! Maintain straight back alignment."
    else:
        return "Exercise type not recognized."
    
def llm_feedback(ex_type, metrics, rule_feedback):
    prompt = f"""
    You are a professional fitness trainer. Analyze the following workout data and give constructive feedback
    on the user's {ex_type} form.

    Data summary:
    {metrics}

    Simple rule-based feedback: "{rule_feedback}"

    Now, in natural language, give specific advice for improvement and encouragement (2-3 sentences).
    """

    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(" Gemini API error:", e)
        return f"(LLM unavailable) {rule_feedback}"


def process_video(video_path, exercise_type="squat", output_path="annotated.mp4"):
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

                shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * width,
                            lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * height]
                elbow = [lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * width,
                         lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * height]
                wrist = [lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x * width,
                         lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y * height]
                hip = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x * width,
                       lm[mp_pose.PoseLandmark.LEFT_HIP.value].y * height]
                knee = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x * width,
                        lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y * height]
                ankle = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * width,
                         lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * height]

                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                knee_angle = calculate_angle(hip, knee, ankle)
                elbow_angles.append(elbow_angle)
                knee_angles.append(knee_angle)

                #rep counting
                if exercise_type == "squat":
                    if knee_angle > 160:
                        stage = "up"
                    if knee_angle < 100 and stage == "up":
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

    rule_fb = basic_feedback(exercise_type, {"knee": knee_angles, "elbow": elbow_angles})  #for stefan to use best judgement
    llm_fb = llm_feedback(exercise_type, metrics, rule_fb)

    print("\n===== WORKOUT SUMMARY =====")
    print(metrics)
    print("\n===== FEEDBACK =====")
    print(llm_fb)

    # Save feedback to file
    with open("feedback.txt", "w") as f:
        f.write("WORKOUT SUMMARY:\n")
        f.write(str(metrics) + "\n\n")
        f.write("FEEDBACK:\n" + llm_fb)

    print("\nAnnotated video saved to:", output_path)
    print("Feedback saved to: feedback.txt")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input exercise video")
    parser.add_argument("--type", default="squat", help="Exercise type (squat/pushup)")
    args = parser.parse_args()

    process_video(args.video, args.type)
