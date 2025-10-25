## Event name : `screenLoadBrand`

## BDD :
### `brand`
GIVEN That I am seeing any brand screen

OR I clicked on show more on brand pages and more products loaded

THEN `screenLoadBrand` event fires

## FIELDS :

| Bertie Field | Bertie Value | Notes | Required | Adobe Field | Adobe Value / Syntax |
| --- | --- | --- | --- | --- | --- |
| ref | see [AllPages](../All/AllPages.md) | | yes | | |
| pageData.pageTitle | brand:`<brand>` | dynamic portion of url, lowercased, slashes replaced by colon| yes | pageName, v14, v3 | pageData.pageTitle;  |
| | | | yes | c1, v70 | pageData.pageTitle (1st part only) |
| | | | yes | c2, v71 | pageData.pageTitle (1st & second parts) |
| | | | yes | c3, v72 | pageData.pageTitle (1st, 2nd & 3rd parts) |
| pageData.pageType | `brand` | | yes | c4, v73 | pageData.pageType |

**Note:** 
1. pageData values here replace the pageData in AllPages - do not create a new pageData event
2. Load above events with [AllPlpPages](../All/AllPlpPages.md) if the page/paginaton contains product list