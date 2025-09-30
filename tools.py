from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv

load_dotenv()


def main():
    tool = DuckDuckGoSearchRun()
    res = tool.invoke("AI Engineer salary in India")
    print(res)


if __name__ == "__main__":
    main()
