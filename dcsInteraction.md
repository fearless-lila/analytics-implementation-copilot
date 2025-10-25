## Event name : `dcsInteraction`

## BDD :
### `dcs.algorithmic.interaction`
GIVEN that I am on any screen

AND I click on any DCS content  

WHEN the component loads

THEN a `dcsInteraction` event fires

## FIELDS :

| Bertie Field                                            | Bertie Value                                 | Notes                                         | Required | Adobe Field | Adobe Value / Syntax                                          |
|---------------------------------------------------------|----------------------------------------------|-----------------------------------------------|----------|-------------|---------------------------------------------------------------|
| ref                                                     | `dcs.algorithmic.interaction`                |                                               | yes      |             |                                                               |
| contentInteractOp.bertieType                            | contentInteractOp                            |                                               | yes      |             |                                                               |
| contentInteractOp.interactionType                       | `dcs`                                        |                                               | yes      |             |                                                               |
| contentInteractOp.content.componentType                 | `c`                                          |                                               | yes      |             |                                                               |
| contentInteractOp.content.componentLocation             |                                              | Values to be populated from ad server.        | yes      |             |                                                               |
| contentInteractOp.content.pageType                      | see table below                              |                                               | yes      | list2       |                                                               |
| contentInteractOp.content.modulePosition                | `<position of module>`                       | not required for Media banners                | yes      | list2       |                                                               |
| contentInteractOp.content.displayArea                   | see table below                              | grid vs carousel                              | yes      | list2       |                                                               |
| contentInteractOp.content.panel[i].campaignId           | `<tracking id if available or content id>`   |                                               | yes      |             |                                                               |
| contentInteractOp.content.panel[i].posInModule          | `<position within module>`                   |                                               | yes      | list2       |                                                               |
| contentInteractOp.content.panel[i].contentFlag          | `<1 if campaignId uses tracking id, else 0>` |                                               | yes      | list2       |                                                               |
| contentInteractOp.content.panel[i].personalisationModel | `<model version>-<model name>`               | do not set key if product is not personalised | yes      | list2       | see list2 syntax `no` if not personalised                     | 
| contentInteractOp.content.panel[i].primaryAdId          | `<adid>` (OS), `<creativeId>` (GAM)          | Display media only                            |          |             |                                                               |
| contentInteractOp.content.panel[i].secondaryAdId        | `<lineItemId>` (GAM)                         | Display media only                            |          |             |                                                               |
|                                                         |                                              |                                               |          | v90         | `<pers/no>` set to `pers` if this interaction is personalised |
|                                                         |                                              |                                               | yes      | v45         | list2 value                                                   |
|                                                         |                                              |                                               | yes      | e45         |                                                               |
|                                                         |                                              |                                               | yes      | v166        | `dcs`                                                         |  

Note - Adobe call is not required for Media banners

## Reference Tables

### pageType options:

| option           | shorthand code |
|------------------|----------------|
| home             | h              |
| favourites       | f              |
| basket           | tf             |
| search           | sr             |
| super department | sd             |
| department       | d              |
| aisle            | a              |
| shelf            | s              |
| buylist          | bl             |
| zone             | z              |
| orders           | o              |
| special offers   | so             |
| pdp              | p              |
| checkout         | c              |
| slots            | sl             |


### displayArea options:

| shorthand code | component               | Status   |
|----------------|-------------------------|----------|
| acc            | CustomWidthAccordion    |          |
| bin            | BrandInformation        |          |
| bnb            | Full Basket Banner      |          |
| bnc            | Banner                  |          |
| bnh            | InspirationalHeroBanner |          |
| bnk            | SkinnyBanner            |          |
| bnn            | Notification Banner     |          |
| bnp            | Proposition Banner      |          |
| bns            | Slim Trade Banner       |          |
| btn            | Event Button            |          |
| csl            | Carousel                |          |
| dfp            | DFP                     |          |
| ftr            | Footer                  |          |
| grd            | Plp grid                |          |
| hdr            | Header                  |          |
| inf            | Info Text               |          |
| isc            | SocialIcons             |          |
| ism            | IconStamp               |          |
| lnk            | Link                    |          |
| nav            | Secondary Navigation    |          |
| sam            | StandaloneMedia         |          |
| sch            | Section Heading         |          |
| smo            | Trade Stamp Offer       |          |
| smp            | Trade Stamp             |          |
| tlp            | Title and Paragraph     |          |
| ttb            | Trade Tile Basic        |          |
| tth            | Trade Tile Horizontal   |          |
| ttl            | Content Title           |          |
| ttv            | Trade Tile Vertical     |          |
| ukn            | Other / Unknown         |          |
| vid            | Trade Tile Video        |          |
| pending        | ContentHeading          |          |
| pending        | ContentStamp            |          |
| bnr            |                         | obsolete |
| div            |                         | obsolete |
| pro            |                         | obsolete |
| ptl            |                         | obsolete |
| sld            |                         | obsolete |
