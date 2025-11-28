from analyze.analyze import Analyze

def main():
    analyze = Analyze("config.yaml")
    analyze.run(r'/Volumes/ADATA SE880/DFL Bundesliga Data Shootout/train/A1606b0e6_0/A1606b0e6_0 (54).mp4')
    #analyze.run("/Volumes/ADATA SE880/DFL Bundesliga Data Shootout/test/test (2).mp4") #mac
    #analyze.run(r"C:\Users\Windows 10\Downloads\DFL Bundesliga Data Shootout\test\test (2).mp4")  #windows

if __name__ == "__main__":
    main()
