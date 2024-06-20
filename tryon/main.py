import cv2
import mediapipe as mp


shirt_images = ['tshirt.png', '1.png', '2.png']
current_shirt_index = 0

def detect_poses_in_video(video_path):
    global current_shirt_index, shirt_images

    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    
    cap = cv2.VideoCapture(video_path)

    
    cv2.namedWindow('Pose Detection and Image Overlay')

   
    def mouse_click_event(event, x, y, flags, params):
        global current_shirt_index

        
        if event == cv2.EVENT_LBUTTONDOWN:
            frame_width = params[0]
            frame_height = params[1]

           
            left_button_area = [(0, 0), (frame_width // 2, frame_height)]
            right_button_area = [(frame_width // 2, 0), (frame_width, frame_height)]

            
            if left_button_area[0][0] <= x <= left_button_area[1][0] and left_button_area[0][1] <= y <= left_button_area[1][1]:
                current_shirt_index = (current_shirt_index - 1) % len(shirt_images)
                print("Shirt image changed to:", shirt_images[current_shirt_index])

           
            if right_button_area[0][0] <= x <= right_button_area[1][0] and right_button_area[0][1] <= y <= right_button_area[1][1]:
                current_shirt_index = (current_shirt_index + 1) % len(shirt_images)
                print("Shirt image changed to:", shirt_images[current_shirt_index])

   
    while cap.isOpened():
        
        ret, frame = cap.read()
        if not ret:
            break

        
        frame_height, frame_width, _ = frame.shape

        
        cv2.setMouseCallback('Pose Detection and Image Overlay', mouse_click_event, [frame_width, frame_height])

        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        
        results = pose.process(frame_rgb)

        
        if results.pose_landmarks:
            
            left_shoulder = (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * frame.shape[1]),
                             int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame.shape[0]))
            right_shoulder = (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame.shape[1]),
                              int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame.shape[0]))
            neck = (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * frame.shape[1]),
                    int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * frame.shape[0]))

            
            shirt_image = shirt_images[current_shirt_index]
            shirt_img = cv2.imread(shirt_image, cv2.IMREAD_UNCHANGED)

            if shirt_img is None:
                print(f"Failed to load image: {shirt_image}")
                continue

           
            shirt_width = abs(right_shoulder[0] - left_shoulder[0]) * 2

            
            shirt_resized = cv2.resize(shirt_img, (int(shirt_width * 0.75), int(shirt_img.shape[0] * shirt_width / shirt_img.shape[1] * 0.7)))

            
            offset = 30  
            overlay_pos_x = (left_shoulder[0] + right_shoulder[0]) // 2 - shirt_resized.shape[1] // 2
            overlay_pos_y = max(1, neck[1] + offset)

            
            overlay_pos_x = max(0, min(overlay_pos_x, frame.shape[1] - shirt_resized.shape[1]))
            overlay_pos_y = max(0, min(overlay_pos_y, frame.shape[0] - shirt_resized.shape[0]))

            
            alpha_s = shirt_resized[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            
            for c in range(0, 3):
                frame[overlay_pos_y:overlay_pos_y + shirt_resized.shape[0],
                      overlay_pos_x:overlay_pos_x + shirt_resized.shape[1], c] = (
                          alpha_s * shirt_resized[:, :, c] +
                          alpha_l * frame[overlay_pos_y:overlay_pos_y + shirt_resized.shape[0],
                                          overlay_pos_x:overlay_pos_x + shirt_resized.shape[1], c])

        
        left_button_area = [(0, 0), (frame_width // 2, frame_height)]
        right_button_area = [(frame_width // 2, 0), (frame_width, frame_height)]

        
        cv2.circle(frame, (frame_width // 4, frame_height // 2), 80, (255, 0, 0), -1)
        cv2.putText(frame, '<', (frame_width // 4 - 30, frame_height // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 10)

       
        cv2.circle(frame, (3 * frame_width // 4, frame_height // 2), 80, (255, 0, 0), -1)
        cv2.putText(frame, '>', (3 * frame_width // 4 - 30, frame_height // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 10)

        
        cv2.imshow('Pose Detection and Image Overlay', frame)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
    cap.release()
    cv2.destroyAllWindows()

def main():
    
    video_path = 'sa.mov' 

    
    detect_poses_in_video(video_path)

if __name__ == "__main__":
    main()
