from .utils.reviewloader import ReviewLoader
import os


def main():
    print(os.getcwd())
    rl = ReviewLoader("./data/train/")

    print(rl.get_df().head)


if __name__=="__main__":
    main()


    
