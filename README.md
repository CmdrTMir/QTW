# QTW
QuantumTransportWorkbench: A Python GUI, as interactive toolkit to explore and understand numerical and analytical solutions to transport, includes expectation value plots.

**english version below**

Dieses Projekt dient als Arbeitsumgebung für die Vorbereitung der Masterarbeit  
**"Computational analysis of transport and protected topological states in open quantum systems"**  
und kombiniert mehrere Python‑Module in einer gemeinsamen GUI.

Es wurden folgende Versionen für die Implementierung verwendet:
Python: 3.13.2
Matplotlib: 3.10.0
Tkinter: 8.6
NumPy: 2.2.3
SciPy: 1.15.2
 
Es ist nicht garantiert, dass andere Versionen funktionieren. Um sicherzustellen, dass die GUI ausgeführt werden kann, können Sie die genannten Version in einer virtuellen Umgebung installieren. 
Schritte für die virtuelle Umgebung:
1. Python 3.13.2 installieren (Haken bei "Add to PATH" nicht vergessen)
2. Kommandozeile im Projektordner öffnen
3. py -3.13 -m venv .venv ausführen
4. .venv\Scripts\activate ausführen
5. pip install matplotlib==3.10.0 ausführen
6. andere dependencies installieren (z.B. NumPy, SciPy)
7. GUI mit python SimGUI.py starten
8. Nach getaner Arbeit: deactivate

Die Videowiedergabe innerhalb der GUI erfordert zudem den VLC Media Player. Falls Sie diesen unter keinen Umständen installieren möchten, finden Sie die Videos im Unterordner /videos und können sie mit Ihrem bevorzugten Player separat abspielen.

**English version:**

The following versions were used for the implementation:
Python: 3.13.2
Matplotlib: 3.10.0
Tkinter: 8.6
NumPy: 2.2.3
SciPy: 1.15.2

It is not guaranteed that other versions will work. To ensure that the GUI can be executed, you can install the mentioned versions in a virtual environment.
Steps for the virtual environment:
1. Install Python 3.13.2 (do not forget to check "Add to PATH")
2. Open command line in the project folder
3. Execute py -3.13 -m venv .venv
4. Execute .venv\Scripts\activate
5. Execute pip install matplotlib==3.10.0 
6. Install other dependencies (e.g. NumPy, SciPy)
7. Start the GUI with python SimGUI.py
8. After finishing work: deactivate

Video playback within the GUI also requires the VLC Media Player. If you do not want to install it under any circumstances, you can find the videos in the \videos subfolder and play them separately with your preferred player.