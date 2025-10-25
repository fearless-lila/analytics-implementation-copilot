## BDD :
### `dcs.algorithmic.impression`
GIVEN That I am on any screen

AND DCS content is displayed  

WHEN the component loads

THEN `dcsImpression` event fires

## FIELDS :

| Bertie Field                                               | Bertie Value                                  | Notes                                         | Required | Adobe Field | Adobe Value / Syntax                              |
|------------------------------------------------------------|-----------------------------------------------|-----------------------------------------------|----------|-------------|---------------------------------------------------|
| ref                                                        | `dcs.algorithmic.impression`                  |                                               |          |             |                                                   |
| renderedContentOp.bertieType                               | `renderedContentOp`                           |                                               |          |             |                                                   |
| renderedContentOp.content[i].componentType                 | `c`                                           |                                               | yes      |             |                                                   |
| renderedContentOp.content[i].componentLocation             |                                               | Values to be populated from ad server.        | yes      |             |                                                   |
| renderedContentOp.content[i].pageType                      | (see shorthand code from table below)         |                                               |          |             |                                                   |
| renderedContentOp.content[i].displayArea                   | (see shorthand code from table below)         |                                               |          |             |                                                   |
| renderedContentOp.content[i].modulePosition                | `<relative position of module>`               | not required for Media banners                |          |             |                                                   |
| renderedContentOp.content[i].panel[i].campaignId           | `<tracking id if available, else content id>` |                                               |          |             |                                                   |
| renderedContentOp.content[i].panel[i].posInModule          | `<position within module>`                    |                                               |          |             |                                                   |
| renderedContentOp.content[i].panel[i].contentFlag          | `<1 if campaignId uses tracking id, else 0>`  |                                               |          |             |                                                   |
| renderedContentOp.content[i].panel[i].personalisationModel | `<model version>-<model name>`                | do not set key if product is not personalised | yes      | list2       | see list2 syntax, `no` if not personalised        | 
| renderedContentOp.content[i].panel[i].primaryAdId          | `<adid>` (OS), `<creativeId>` (GAM)           | Display media only                            |          |             |                                                   |
| renderedContentOp.content[i].panel[i].secondaryAdId        | `<lineItemId>` (GAM)                          | Display media only                            |          |             |                                                   |
|                                                            |                                               |                                               |          | v90         | `<pers/no>` set to `pers` if any are personalised |
|                                                            |                                               |                                               |          | e167        |                                                   |

Note - No Adobe event required for Media banners.

### pageType options:

| option | shorthand code |
| --- | --- |
|home|h|
|favourites|f|
|basket|tf|
|search|sr|
|super department|sd|
|department|d|
|aisle|a|
|shelf|s|
|buylist|bl|
|zone|z|
|orders|o|
|special offers|so|
|pdp|p|
|checkout|c|
|slots|sl|


### displayArea options:

| shorthand code | component | Status |
| --- | --- | --- |
|	acc	|	CustomWidthAccordion	||
|	bin	|	BrandInformation	||
|	bnb	|	Full Basket Banner	||
|	bnc	|	Banner	||
|	bnh	|	InspirationalHeroBanner	||
|	bnk	|	SkinnyBanner	||
|	bnn	|	Notification Banner	||
|	bnp	|	Proposition Banner	||
|	bns	|	Slim Trade Banner	||
|	btn	|	Event Button	||
|	csl	|	Carousel	||
|	dfp	|	DFP	||
|	ftr	|	Footer	||
|	grd	|	Plp grid	||
|	hdr	|	Header	||
|	inf	|	Info Text	||
|	isc	|	SocialIcons	||
|	ism	|	IconStamp	||
|	lnk	|	Link	||
|	nav	|	Secondary Navigation	||
|	sam	|	StandaloneMedia	||
|	sch	|	Section Heading	||
|	smo	|	Trade Stamp Offer	||
|	smp	|	Trade Stamp	||
|	tlp	|	Title and Paragraph	||
|	ttb	|	Trade Tile Basic	||
|	tth	|	Trade Tile Horizontal	||
|	ttl	|	Content Title	||
|	ttv	|	Trade Tile Vertical	||
|	ukn	|	Other / Unknown	||
|	vid	|	Trade Tile Video	||
|	pending	|	ContentHeading	||
|	pending	|	ContentStamp	||
|	bnr	|	|obsolete	|
|	div	|	|obsolete	|
|	pro	|	|obsolete	|
|	ptl	|	|obsolete	|
|	sld	|	|obsolete	|
