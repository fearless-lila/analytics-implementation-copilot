import requests

# Replace with your token and repo
token = "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"  # or leave empty if repo is public
repo = "fearless-lila/analytics-implementation-copilot"
query = "basketSort.md"  # a file or keyword you know exists

url = f"https://api.github.com/search/code?q={query}+repo:{repo}"
headers = {"Authorization": f"token {token}"} if token else {}

response = requests.get(url, headers=headers)

print("HTTP status code:", response.status_code)
if response.status_code != 200:
    print("Error:", response.text)
else:
    results = response.json().get("items", [])
    print(f"Number of results found: {len(results)}")
    for item in results[:5]:
        print(f"{item['name']} - {item['html_url']}")
