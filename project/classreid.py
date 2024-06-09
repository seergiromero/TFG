import cv2
import numpy as np
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from torchreid.reid.utils import FeatureExtractor


class reidentification:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.extractor = FeatureExtractor(
            model_name='osnet_x0_75',
            model_path='osnet_x0_75_market.pth',
            device='cpu'
        )
        self.feature_history = {}
        self.temp_features = {}
        

    def extract_features_from_roi(self, roi):
       
        if roi.shape[0] > 0 and roi.shape[1] > 0:
            try:

                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

                roi_tensor = self.transform(roi)
                roi_tensor = roi_tensor.unsqueeze(0)

                features = self.extractor(roi_tensor)

                return features.squeeze().numpy()
            except Exception as e:
                print(f"Error al procesar la ROI: {e}")
                
        return None  

    def reidentification(self, features):

        similarity_threshold = 0.70  
        found_existing_id = False
        best_match_id = None
        best_similarity = similarity_threshold 

        for stored_id, stored_features_list in self.feature_history.items():
            similarities = [float(cosine_similarity([features], [stored_features])) for stored_features in stored_features_list]
            avg_similarity = np.mean(similarities) 
            if avg_similarity > best_similarity:
                best_similarity = avg_similarity
                best_match_id = stored_id

        if best_match_id is not None:
            found_existing_id = True

        else:
            best_similarity = 0
            best_match_id = 0

        return found_existing_id, best_match_id, best_similarity



    def reidentification_process(self, features, track_id, tracker, count):

        
        if not self.feature_history:
            new_track_id = 1
            tracker.update_tracks(new_track_id)
            self.update_feature_history(new_track_id, features)
    
        else:
            if track_id not in self.temp_features: 
                self.temp_features[track_id] = []
                tracker.update_tracks(0, track_id) 
            self.temp_features[track_id].append(features)

            if len(self.temp_features[track_id]) >=  count:
                consistent_matches = []
                for features in self.temp_features[track_id]:
                    _, verified_id, verified_similarity = self.reidentification(features)
                    consistent_matches.append((verified_id, verified_similarity, features))
                
                best_match_counts = {}
                for match in consistent_matches:
                    if match[0] in best_match_counts:
                        best_match_counts[match[0]] += 1
                    else:
                        best_match_counts[match[0]] = 1

                max_count_id = max(best_match_counts, key=best_match_counts.get)
                max_count = best_match_counts[max_count_id]

                if max_count > count / 2 and max_count_id != 0:
                    tracker.update_tracks(max_count_id, track_id)

                    for verified_id, verified_similarity, features in consistent_matches:
                        if verified_id == max_count_id:
                            self.update_feature_history(max_count_id, features)
                else:
                    new_track_id = max(self.feature_history.keys()) + 1
                    self.update_feature_history(new_track_id, features)
                    tracker.update_tracks(new_track_id, track_id)

                del self.temp_features[track_id]

    def update_feature_history(self, track_id, features):
        
        if track_id in self.feature_history:
            self.feature_history[track_id].append(features)
            if len(self.feature_history[track_id]) > 25:  
                self.feature_history[track_id] = self.feature_history[track_id][-25:]
        else:
            self.feature_history[track_id] = [features]