## Event name : `screenLoadHome`

## BDD :
### `home`
GIVEN That I am seeing the home screen

THEN `screenLoadHome` event fires

## FIELDS :

| Bertie Field | Bertie Value | Notes | Required | Adobe Field | Adobe Value / Syntax |
| --- | --- | --- | --- | --- | --- |
| ref | see [AllPages](../All/AllPages.md) | | yes | | |
| pageData.pageTitle | `shop homepage` | | yes | pageName, v14, v3 | pageData.pageTitle |
| | | | yes | c1, v70 | pageData.pageTitle (1st part only) |
| | | | yes | c2, v71 | pageData.pageTitle (1st & second parts) |
| | | | yes | c3, v72 | pageData.pageTitle (1st, 2nd & 3rd parts) |
| pageData.pageType | `shop homepage` | | yes | c4, v73 | pageData.pageType |

Note pageData values here replace the pageData in AllPages - do not create a new pageData event
