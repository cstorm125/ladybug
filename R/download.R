input_path = '../data/'

ladyboy_df = readRDS(paste0(input_path,'ladyboy_df.rds')) %>% filter(age <= 40)
write.table(ladyboy_df$smallPic,'ladyboy_smallPic.txt',row.names = FALSE)

for (i in 1:dim(ladyboy_df)[1]){
  download.file(url=as.character(ladyboy_df[i,]$smallPic),
                destfile = paste0('../data/ladyboy/',ladyboy_df[i,]$username,'.jpg'))
}



ladyboy_big = readRDS(paste0(input_path,'ladyboy_big.rds'))
write.table(ladyboy_big$value,'ladyboy_bigPic.txt',row.names = FALSE)

for (i in 16022:dim(ladyboy_big)[1]){
  download.file(url=as.character(ladyboy_big[i,]$value),
  destfile = paste0('../data/ladyboy_big/',ladyboy_big[i,]$username,'_',ladyboy_big[i,]$variable,'.jpg'))
}

girl_df = readRDS(paste0(input_path,'girl_df.rds'))
write.table(girl_df$smallPic,'girl_smallPic.txt',row.names = FALSE)

for (i in 1:dim(girl_df)[1]){
  download.file(url=as.character(girl_df[i,]$smallPic),
                destfile = paste0('../data/girl/',girl_df[i,]$username,'.jpg'))
}
