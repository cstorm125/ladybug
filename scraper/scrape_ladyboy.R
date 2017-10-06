les_packages = lapply(c("stringi","RSelenium","rvest","data.table","ggplot2","tidyr","dplyr"), 
                       require, character.only = TRUE)

#sudo docker run -d -p 4445:4444 selenium/standalone-firefox:2.53.0
#docker stop $(docker ps -q)
remDr <- remoteDriver(port = 4445L)
remDr$open()

#login
username = 'cebril@gmail.com'
password = 'kha07anz'

#ladyboy dataset
#go to home page
remDr$navigate("https://myladyboydate.com/")

#click login
loginButton = remDr$findElement(using = "css", "[class = 'btn btn-primary btn-login']")
loginButton$sendKeysToElement(list('\uE007'))
remDr$getTitle()
#enter user and pass
userText = remDr$findElement(using = "css", "[name = 'email']")
userText$sendKeysToElement(list(username))
passText = remDr$findElement(using = "css", "[name = 'password']")
passText$sendKeysToElement(list(password))
signInButton = remDr$findElement(using = "css", "[id = 'submit-button']")
signInButton$sendKeysToElement(list('\uE007'))
remDr$getTitle()

#search
remDr$navigate("https://myladyboydate.com/search/")
remDr$getTitle()
#gender
genderOption = remDr$findElement(using = "css", "[name = 'gender']")
genderOption$getElementAttribute('value')
#from
fromOption = remDr$findElement(using = "css", "[name = 'country']")
fromOption$getElementAttribute('value')
#checked with profile pic
checked = remDr$findElement(using = "css", "[id = 'photo-checkbox']")
checked$getElementAttribute('value')


#loop through search
total_pages = 0:499
pages = paste0('https://myladyboydate.com/search/?page=',total_pages,'&low=0')

url_list = NULL
smallPic_list = NULL
user_list = NULL
detail_list = NULL
country_list = NULL

for (page in pages){
  print(c)
  c = c+1
  page = page
  #catalog
  remDr$navigate(page)
  profileList = remDr$findElement(using = "css", "[class = 'row profiles-list loading']")
  profileDIV=profileList$findChildElements(using = "xpath",'*')
  for(i in 1:length(profileDIV)){
    #profile URL
    profileURL = profileDIV[[i]]$findChildElement(using='css',"[class='avatar']")
    url_list = c(url_list,profileURL$getElementAttribute('href')[[1]])
    
    #small pic
    smallPic = profileURL$findChildElement(using = "xpath",'img')
    smallPic_list = c(smallPic_list,smallPic$getElementAttribute('src')[[1]])
    
    #user_list
    user = profileDIV[[i]]$findChildElement(using='css',"[class='username']")
    the_user = user$findChildElement(using='tag',"a")
    user_list = c(user_list,the_user$getElementText()[[1]])
    
    #detail_list
    detail = profileDIV[[i]]$findChildElement(using='css',"[class='details']")
    detail_list = c(detail_list,detail$getElementText()[[1]])
    
    #country_list
    location = profileDIV[[i]]$findChildElement(using='css',"[class='location']")
    country = location$findChildElement(using='xpath','span')
    country_list = c(country_list,country$getElementAttribute('title')[[1]])
  }
}

ladyboy_df = data.frame(url=url_list,
                       smallPic=smallPic_list,
                       user=user_list,
                       detail=detail_list,
                       country=country_list,
                       stringsAsFactors=FALSE)

ladyboy_df = ladyboy_df %>% 
  mutate(username = sapply(url, FUN=function(x) substring(x,36)),
         age= sapply(ladyboy_df$detail, FUN=function(x) strsplit(x,' / ')[[1]][1] %>% as.numeric),
         country = tolower(country)) %>% unique

saveRDS(ladyboy_df,'ladyboy_df.rds')
ladyboy_df = readRDS('ladyboy_df.rds')

bigPic_list = list()
dim(ladyboy_df)[1]

#get more details
for(i in 1:dim(ladyboy_df)[1]){
  print(i)
  tryCatch({
  remDr$navigate(paste0(ladyboy_df[i,'url'],'/photos'))
  rowPhotos = remDr$findElement(using = "css", "[class = 'row photos-list']")
  photos = rowPhotos$findChildElements(using='tag','img')
  bigPic_list[[substring(ladyboy_df[i,'url'],36)]] = sapply(photos,function(x) x$getElementAttribute('src')[[1]])},
  error=function(err) print(err))
}

ladyboy_big = as.data.frame(t(stri_list2matrix(bigPic_list)))
names(ladyboy_big) = 1:dim(ladyboy_big)[2]
ladyboy_big$username = names(bigPic_list)
ladyboy_big = melt(ladyboy_big,id.var='username') %>% filter(!is.na(value)) %>% arrange(username)
saveRDS(ladyboy_big,'ladyboy_big.rds')
ladyboy_big = readRDS('ladyboy_big.rds')

#date in asia
#search
smallPic_list = NULL
detail_list = NULL

remDr$navigate(paste0('https://www.dateinasia.com/Search.aspx?pg=1&g=2&af=18&at=40'))

for(i in 1:1000){
  tryCatch({
    webElem <- remDr$findElement("css", "body")
    webElem$sendKeysToElement(list(key = "end"))
    #whole grid
    searchgrid = remDr$findElement(using = "css", "[class = 'searchgrid']")
    subA = searchgrid$findChildElements('tag','a')

    #detail
    the_detail = sapply(subA,FUN=function(x) unlist(x$getElementText()))
    detail_list = c(detail_list,the_detail)
    
    imageContain = sapply(subA,FUN=function(x) unlist(x$findChildElement('css','span > span > img')))
    the_smallPic = sapply(imageContain,FUN=function(x) unlist(x$getElementAttribute('src')))
    smallPic_list = c(smallPic_list,the_smallPic)
    
    
    nextButton = remDr$findElement(using = "css", "[class = 'bttnf mls mrs']")
    nextButton$sendKeysToElement(list('\uE007'))
    
    print(paste0('Success for ',i))
  },
  error=function(err) print(err))
}

name_age = sapply(detail_list,FUN=function(x) strsplit(as.character(x),', ')[[1]][1])
age_list=sapply(name_age,FUN=function(x) {
  y=strsplit(x,'\n')[[1]]
  as.numeric(y[length(y)])
  })
user_list = sapply(name_age,FUN=function(x) {
  y=strsplit(x,'\n')[[1]]
  tolower(y[length(y)-1])
})
country_list = sapply(detail_list,FUN=function(x){
  y=strsplit(as.character(x),', ')[[1]][2]
  tolower(strsplit(y,'\n')[[1]][2])
  })

girl_df = data.frame(detail=detail_list,
                     smallPic=smallPic_list,
                     username=user_list,
                     age=age_list,
                     country=country_list) %>% unique 

saveRDS(girl_df,'girl_df.rds')
girl_df = readRDS('girl_df.rds')

#close
remDr$close()

# 
# #asian friend finder
# #go to home page
# remDr$navigate("http://asianfriendfinder.com/p/memsearch.cgi")
# remDr$getTitle()
# 
# #enter user and pass
# userText = remDr$findElement(using = "css", "[id = 'login_handle']")
# userText$sendKeysToElement(list(username))
# passText = remDr$findElement(using = "css", "[id = 'login_password']")
# passText$sendKeysToElement(list(password))
# signInButton = remDr$findElement(using = "css", "[id = 'cover_login_button']")
# signInButton$sendKeysToElement(list('\uE007'))
# 
# #search
# smallPic_list = NULL
# user_list = NULL
# age_list = NULL
# country_list = NULL
# 
# remDr$navigate(paste0('https://asiafriendfinder.com/go/page/new_search.html'))
# savedSearch = remDr$findElement(using = "css", "[id = 'saved-search-0']")
# savedSearch$sendKeysToElement(list('\uE007'))
# 
# for(i in 1:300){
#   tryCatch({
#     
#     #pic
#     smallPic = remDr$findElements(using = "css", "[class = 'member_cell_image_container']")
#     the_smallPic = sapply(smallPic,FUN=function(x) unlist(x$findChildElement('tag','img')$getElementAttribute('src')))
#     the_smallPic = the_smallPic[!grepl('square',the_smallPic)]
#     smallPic_length = length(the_smallPic)
#     print(smallPic_length)
#     
#     #user
#     ladyname = remDr$findElements(using = "css", "[data-ga-event = 'Member username:Text']")
#     the_user = unlist(sapply(ladyname,FUN=function(x) x$getElementText()))
#     user_length = length(ladyname)
#     print(user_length)
#     
#     #age
#     age = remDr$findElements(using = "css", "[class = 'member_cell_info_row1']")
#     the_age = unlist(sapply(age,FUN=function(x) x$getElementText()))
#     the_age = the_age[2:21]
#     age_length = length(the_age)
#     print(age_length)
#     
#     #country
#     country = remDr$findElements(using = "css", "[class = 'location muted']")
#     the_country = unlist(sapply(country,FUN=function(x) x$getElementText()))
#     country_length= length(the_country)
#     print(country_length)
#     
#     #count check
#     if(smallPic_length != user_length | country_length != user_length | age_length != user_length) {
#       print(paste0('Page ',i,' does not have matching numbers'))
#       next}
#     
#     
#     smallPic_list = c(smallPic_list,the_smallPic)
#     user_list = c(user_list,the_user)
#     age_list = c(age_list,the_age)
#     country_list = c(country_list,the_country)
#     
#     
#     nextButton = remDr$findElement(using = "css", "[id = 'next']")
#     nextButton$sendKeysToElement(list('\uE007'))
#     
#     print(paste0('Success for ',i))
#   },
#   error=function(err) print(err))
# }
# 
# girl_df = data.frame(user=user_list,
#                      smallPic=smallPic_list,
#                      age_sex=age_list,
#                      location=tolower(country_list),
#                      stringsAsFactors=FALSE) %>%
#   mutate(country=sapply(location,FUN=function(x) strsplit(x,', ')[[1]][2]),
#          age=sapply(age_sex,FUN=function(x) substr(strsplit(x,'\n')[[1]][2],1,2)) %>% as.numeric)
# 
# 
# saveRDS(ladyboy_df,'girl_df.rds')
# girl_df = readRDS('girl_df.rds')
