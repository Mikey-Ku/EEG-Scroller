"""Open the EEG Scroller app (macOS).

Run: python scripts/app.py          (starts muselsl itself, or uses a stream already running)
     python scripts/app.py --sim    (simulated EEG, no headset)
"""

from eegscroll.gui.app import main

if __name__ == "__main__":
    main()
