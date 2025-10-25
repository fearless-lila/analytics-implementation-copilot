## Event name : `screenLoadFavourites`

## BDD :
### `favourites`
GIVEN That I am seeing any Favourites list screen

THEN `screenLoadFavourites` event fires

## FIELDS :

| Bertie Field | Bertie Value | Notes | Required | Adobe Field | Adobe Value / Syntax |
| --- | --- | --- | --- | --- | --- |
| ref | see [AllPages](../All/AllPages.md) | | yes | | |
| pageData.pageTitle | `favourites:`<`my favourites`/`my usuals`/`last order`> | | yes | pageName, v14, v3 | pageData.pageTitle |
| | | | yes | c1, v70 | pageData.pageTitle (1st part only) |
| | | | yes | c2, v71 | pageData.pageTitle (1st & second parts) |
| | | | yes | c3, v72 | pageData.pageTitle (1st, 2nd & 3rd parts) |
| | | | yes | c4, v73 | pageData.pageType |

**Note:** pageData values here replace the pageData in AllPages - do not create a new pageData event
