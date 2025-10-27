import requests

token = "" # need to get this from notes
repo = "fearless-lila/analytics-implementation-copilot"
query = "basketSort"

headers = {"Authorization": f"token {token}"} if token else {}

# 1️⃣ Search by filename
url_filename = f"https://api.github.com/search/code?q=filename:{query}+repo:{repo}"
r1 = requests.get(url_filename, headers=headers)
results = r1.json().get("items", [])
print(f"Filename search: {len(results)} results")

# 2️⃣ If none found, search by content
if not results:
    url_content = f"https://api.github.com/search/code?q={query}+repo:{repo}"
    r2 = requests.get(url_content, headers=headers)
    results = r2.json().get("items", [])
    print(f"Content search: {len(results)} results")

# 3️⃣ Display results
for item in results[:5]:
    print(f"{item['name']} - {item['html_url']}")
