import cv2
import numpy as np
import mediapipe as mp
from statistics import mean

print("working")


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

import google.generativeai as genai
genai.configure(api_key="AIzaSyBBv9qZyzwaXRSR1m0yPkpvYZSKFhdNP80")


from threading import Thread

class VideoStream:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        Thread(target=self.update, daemon=True).start()
    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.cap.read()
    def read(self):
        return self.frame.copy() if self.ret else None
    def stop(self):
        self.stopped = True
        self.cap.release()


import csv
import os
from datetime import datetime

CSV_FILE = "workout_history.csv"

def append_workout_to_csv(metrics, exercise_type, rule_feedback):
    header = ["timestamp", "exercise_type", "total_reps", "avg_knee_angle", 
              "min_knee_angle", "avg_elbow_angle", "min_elbow_angle", "rule_feedback_summary"]

    feedback_summary = ";".join(rule_feedback)
    row = [datetime.now().isoformat(), exercise_type, metrics["total_reps"],
           metrics.get("avg_knee_angle"), metrics.get("min_knee_angle"),
           metrics.get("avg_elbow_angle"), metrics.get("min_elbow_angle"),
           feedback_summary]

    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


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




#rules for - Stefan
# def basic_feedback(ex_type, angles):
#     if ex_type == "squat":
#         knee_angles = angles["knee"]
#         avg_knee = mean(knee_angles)
#         min_knee = min(knee_angles)
#         if min_knee > 100:
#             return f"Your squats seem shallow (min knee angle ~{int(min_knee)}°). Try going deeper."
#         elif avg_knee < 80:
#             return "Nice depth, keep maintaining that squat form!"
#         else:
#             return "Good effort! Maintain consistent squat depth."
#     elif ex_type == "pushup":
#         elbow_angles = angles["elbow"]
#         min_elbow = min(elbow_angles)
#         if min_elbow > 110:
#             return f"Try lowering yourself more in pushups (min elbow angle ~{int(min_elbow)}°)."
#         else:
#             return "Solid pushup depth! Maintain straight back alignment."
#     else:
#         return "Exercise type not recognized."

def basic_feedback(ex_type, angles):
    feedback_tags = []


    if ex_type == "squat":
        knee_angles = angles["knee"]
        avg_knee = mean(knee_angles)
        min_knee = min(knee_angles)
        # if min_knee > 100:
        #     feedback_tags.append("shallow_squat")
        # if avg_knee < 80:
        #     feedback_tags.append("good_depth")
        # if 80 <= avg_knee <= 100:
        #     feedback_tags.append("maintain_depth")

        # Depth
        if min_knee > 120:
            feedback_tags.append("shallow_squat")
        elif min_knee < 80:
            feedback_tags.append("deep_squat")
        else:
            feedback_tags.append("good_depth")


        
        # General form
        if avg_knee < 90:
            feedback_tags.append("strong_squat")
        elif avg_knee > 110:
            feedback_tags.append("needs_more_control")


    elif ex_type == "pushup":
        elbow_angles = angles["elbow"]
        min_elbow = min(elbow_angles)
        if min_elbow > 110:
            feedback_tags.append("lower_body_more")
        else:
            feedback_tags.append("good_pushup_form")


    else:
        feedback_tags.append("exercise_not_recognized")
    return feedback_tags

    
def llm_feedback(ex_type, metrics, rule_feedback):
    prompt = f"""
    You are a professional fitness trainer. Analyze the following workout data and give constructive feedback
    on the user's {ex_type} form.

    Data summary:
    {metrics}

    Simple rule-based feedback: "{rule_feedback}"

    Now, in natural language, give specific advice for improvement and encouragement (1-2 sentences).
    """

    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash-lite")
        # response = model.generate_content(prompt)
        # return response.text.strip()
        response = model.generate_content(
            prompt,
            stream=True
        )

        text = ""
        for chunk in response:
            if chunk.text:
                text += chunk.text

        return text.strip()


    except Exception as e:
        print(" Gemini API error:", e)
        return f"(LLM unavailable) {rule_feedback}"

def get_historical_feedback(exercise_type):
    history = []
    if not os.path.isfile(CSV_FILE):
        return history
    with open(CSV_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["exercise_type"] == exercise_type:
                history.append(row)
    return history

def llm_feedback_with_history(ex_type, metrics, rule_feedback):
    history = get_historical_feedback(ex_type)
    
    history_text = "\n".join([f"Reps: {h['total_reps']}, Feedback: {h['rule_feedback_summary']}" for h in history[-10:]])
    rule_summary = ";".join(rule_feedback)

    prompt = f"""
    You are a professional fitness trainer. Analyze the user's workout data.

    Current session metrics:
    {metrics}

    Current session rule-based feedback: {rule_summary}

    Historical last 10 sessions:
    {history_text}

    Output in natural language:
    1) One thing the user did well this session.
    2) One thing to improve this session.
    3) One particular column/metric the user has been doing very well over the course of history and how it has improved percentise wise. THe percent improvement or reduction is a must have
    Provide extremely concise and actionable advice in approx 20 words. Dont focus a lot on grammar and stuff, be extremely to the point
    """

    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash-lite")
        response = model.generate_content(prompt, stream=True)
        text = ""
        for chunk in response:
            if chunk.text:
                text += chunk.text
        return text.strip()
    except Exception as e:
        print("Gemini API error:", e)
        return f"(LLM unavailable) {rule_summary}"


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
    # width, height = 640, 480



    # out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (width, height))
    
    # Metrics
    # knee_angles, elbow_angles = [], []
    knee_angles, elbow_angles, hip_angles, shoulder_angles = [], [], [], []

    reps, stage = 0, None
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
        # while True:
            ret, frame = cap.read()

            """
            gotta do this for multi threading:
            """
            # vs = VideoStream(VIDEO_PATH)
            # frame = vs.read()

            # frame = cv2.resize(frame, (640, 480))


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



                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                knee_angle = calculate_angle(hip, knee, ankle)


                # Compute additional angles
                hip_angle = calculate_angle(shoulder, hip, knee)       # For squat posture
                shoulder_angle = calculate_angle(elbow, shoulder, hip) # For pushup/back alignment


                # elbow_angles.append(elbow_angle)
                # knee_angles.append(knee_angle)


                # Append all to their respective lists
                elbow_angles.append(elbow_angle)
                knee_angles.append(knee_angle)
                hip_angles.append(hip_angle)
                shoulder_angles.append(shoulder_angle)

                #rep counting
                # if exercise_type == "squat":
                #     if knee_angle > 150:
                #         stage = "up"
                #     if knee_angle < 125 and stage == "up":
                #         stage = "down"
                #         reps += 1
                if exercise_type == "squat":
                    if knee_angle > 160:
                        stage = "up"
                    if knee_angle < 140 and stage == "up":
                        stage = "down"
                        reps += 1

                elif exercise_type == "pushup":
                    if elbow_angle > 160:
                        stage = "up"
                    if elbow_angle < 90 and stage == "up":
                        stage = "down"
                        reps += 1

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

    # rule_fb = basic_feedback(exercise_type, {"knee": knee_angles, "elbow": elbow_angles})
    rule_fb = basic_feedback(exercise_type, {
        "knee": knee_angles,
        "elbow": elbow_angles,
        "hip": hip_angles,
        "shoulder": shoulder_angles
    })
    append_workout_to_csv(metrics, exercise_type, rule_fb)
    llm_fb = llm_feedback_with_history(exercise_type, metrics, rule_fb)


    # rule_fb = basic_feedback(exercise_type, {"knee": knee_angles, "elbow": elbow_angles})  #for stefan to use best judgement
    # llm_fb = llm_feedback(exercise_type, metrics, rule_fb)

    # print("\n===== WORKOUT SUMMARY =====")
    # print(metrics)
    # print("\n===== FEEDBACK =====")
    # print(llm_fb)

    # Save feedback to file
    with open("feedback.txt", "w") as f:
        f.write("WORKOUT SUMMARY:\n")
        f.write(str(metrics) + "\n\n")
        f.write("FEEDBACK:\n" + llm_fb)




    print(metrics)


    # Save feedback to file
    # with open("feedback.txt", "w") as f:
    #     f.write("WORKOUT SUMMARY:\n")
    #     f.write(str(metrics) + "\n\n")
    #     f.write("FEEDBACK:\n" + llm_fb)

    print("\nAnnotated video saved to:", output_path)
    print("Feedback saved to: feedback.txt")

    cv2.destroyAllWindows()

if __name__ == "__main__":


    url = "http://10.32.81.159:8080/video"  # IP Webcam app
    # cap = cv2.VideoCapture(url)

    
    # ============================
    # Hardcoded settings (edit here)
    # ============================
    VIDEO_PATH = "stefan.mp4"   # <- change this
    # VIDEO_PATH= "d3.mp4"
    # VIDEO_PATH = url
    EXERCISE_TYPE = "squat"                   # <- squat / pushup / etc.
    OUTPUT_PATH = "annotated_output_ananya2.mp4"      # <- change if needed
    # ============================

    process_video(VIDEO_PATH, EXERCISE_TYPE, OUTPUT_PATH)
