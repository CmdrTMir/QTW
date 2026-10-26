import os
import sys
import vlc
import tkinter as tk

class VideoPlayer:
    def __init__(self, container, video_name):
        self.container = container

        script_dir = os.path.dirname(os.path.abspath(__file__))
        video_rel_path = os.path.join("videos", video_name)
        video_abs_path = os.path.join(script_dir, video_rel_path)

        print(f"Video-Pfad: {video_abs_path}")
        print(f"Datei existiert: {os.path.exists(video_abs_path)}")
        # VLC-Instanz erstellen
        try:
            self.vlc_instance = vlc.Instance("--no-xlib", "--vout=direct3d9", "--quiet")
            self.player = self.vlc_instance.media_player_new()
            if sys.platform == "win32":
                self.player.set_hwnd(self.container.winfo_id())
            elif sys.platform.startswith("linux"):
                self.player.set_xwindow(self.container.winfo_id())
            elif sys.platform == "darwin":
                self.player.set_nsobject(self.container.winfo_id())

            media = self.vlc_instance.media_new(video_abs_path)
            self.player.set_media(media)
            #self.player.play()

        except Exception as e:
            print(f"Fehler beim Initialisieren des Video-Players: {e}")
            self.player = None

    def play(self):
        if self.player is not None:
            self.player.play()
    def pause(self):
        if self.player is not None:
            self.player.pause()
    def stop(self):
        if self.player is not None:
            self.player.stop()

    def cleanup(self):
        # Gibt VLC-Ressourcen frei. Wird beim schließen aufgerufen.
        if self.player is not None:
            self.player.stop()
            self.player.release()
        if hasattr(self, 'vlc_instance'):
            self.vlc_instance.release()

def create_video_controls(parent, player):
    control_frame = tk.Frame(parent)
    control_frame.pack(pady=5)

    tk.Button(control_frame, text="Play", command=player.play, width=10, font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=5)
    tk.Button(control_frame, text="Pause", command=player.pause, width=10, font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=5)
    tk.Button(control_frame, text="Stop", command=player.stop, width=10, font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=5)

    return control_frame
