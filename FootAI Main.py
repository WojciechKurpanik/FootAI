from analyze.analyze import Analyze
from FootAIGUI import FootAIGUI

def main():
    analyze = Analyze("config.yaml")
    #analyze.run("08fd33_0.mp4")
    analyze.run("/Volumes/ADATA SE880/DFL Bundesliga Data Shootout/test/test (2).mp4") #mac
    #analyze.run(r"C:\Users\Windows 10\Downloads\DFL Bundesliga Data Shootout\test\test (2).mp4")  #windows
    # app = FootAIGUI()
    # app.run()


if __name__ == "__main__":
    main()
