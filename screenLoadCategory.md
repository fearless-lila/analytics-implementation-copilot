## BDD :
### `screenLoadCategory`
GIVEN That I am on browse category page

THEN fire screenLoadCategory event with below value

## FIELDS :
| Bertie Field | Bertie Value | Notes | Required | Adobe Field | Adobe Value / Syntax |
| --- | --- | --- | --- | --- | --- |
| ref | see [AllPages](../All/AllPages.md) | | yes | | |
| pageData.pageTitle | `browse:<super department>:<department>:<aisle>:<shelf>` | use as many levels as the page has hierarchy. Do not include 'all' on higher levels or trailing colons | yes | pageName, v14, v3 | pageData.pageTitle |
| | | | yes | c1, v70 | pageData.pageTitle (1st part only) |
| | | | yes | c2, v71 | pageData.pageTitle (1st & second parts) |
| | | | yes | c3, v72 | pageData.pageTitle (1st, 2nd & 3rd parts) |
| pageData.pageType | `<department / ailse / shelf / sub-shelf>`:`hub` | | yes | c4, v73 | pageData.pageType |
| pageData.category[i].catFriendlyLevel | `super department` | | yes | | |
| pageData.category[i].categoryName | `<super department>` |  | yes | | |
| pageData.category[i].catFriendlyLevel | `department` | use as many objects as the page has hierarchy levels | yes | | |
| pageData.category[i].categoryName | `<department>` | | yes | | |
| pageData.category[i].catFriendlyLevel | `aisle` | use as many objects as the page has hierarchy levels | yes | | |
| pageData.category[i].categoryName | `<aisle>` | | yes | | |
| pageData.category[i].catFriendlyLevel | `shelf` | use as many objects as the page has hierarchy levels | yes | | |
| pageData.category[i].categoryName | `<shelf>` | | yes | | |

  Note pageData values here replace the pageData in AllPages - do not create a new pageData event