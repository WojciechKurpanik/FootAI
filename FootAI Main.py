from analyze.analyze_input import Analyze

def main():
    analyze = Analyze("config.yaml")
    analyze.run("/Volumes/ADATA SE880/DFL Bundesliga Data Shootout/test/test (2).mp4")

if __name__ == "__main__":
    main()
