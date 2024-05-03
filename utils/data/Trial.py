from scipy.io import loadmat
from .Labels import Labels

class Trial:
    def __init__(self, path) -> None:
        print(f"Loading Trial {path}")
        self.path = path
        self.data = None
        self.human_id = self.get_human_id()
        self.videos = self.get_videos()
        self.session_id = int(path.split('/')[-2])
        print(f"Loaded list of videos: {self.videos}")

    def load_data(self):
        return loadmat(self.path)

    def get_human_id(self):
        return self.path.split('/')[-1].split('_')[0]
    
    def get_videos(self):
        if self.data is None:
            data = self.load_data()
        else:
            data = self.data
        keys = [key for key in list(data.keys()) if not key.startswith('__')]
        return keys

    def get_video_data(self, video):
        if type(video) == int:
            video = self.videos[video]
        
        if self.data is None:
            self.data = self.load_data()

        return self.data[video]
    
    def __getitem__(self, idx):
        return self.get_video_data(idx), Labels.get_session_labels(self.session_id)[idx]
    
    def __len__(self):
        return len(self.videos)
