CREATE TABLE brands ( code TEXT PRIMARY KEY, -- 銘柄 コード 
name TEXT, -- 銘柄 名（ 正式 名称） 
short_name TEXT, -- 銘柄 名（ 略称）
market TEXT, -- 上場 市場 名 
sector TEXT, -- セクタ 
unit INTEGER -- 単元株 数 
);

CREATE TABLE new_brands ( -- 上場 情報 
code TEXT, -- 銘柄 コード 
date TEXT, -- 上場 日 
PRIMARY KEY( code, date) 
); 

CREATE TABLE delete_brands ( -- 廃止 情報 
code TEXT, -- 銘柄 コード 
date TEXT, -- 廃止 日 
PRIMARY KEY( code, date)
);

CREATE TABLE raw_prices ( 
code TEXT, -- 銘柄 コード 
date TEXT, -- 日付 
open REAL, -- 初値 
high REAL, -- 高値
low REAL, -- 安値 
close REAL, -- 終値 
volume INTEGER, -- 出来高 
PRIMARY KEY( code, date) );

CREATE TABLE divide_union_data ( 
code TEXT, -- 銘柄 コード 
date_of_right_allotment TEXT, -- 権利 確定 日 
before REAL, -- 分割・併合 前 株 数 
after REAL, -- 分割・併合 後 株 数 
PRIMARY KEY( code, date_of_right_allotment) 
);

CREATE TABLE prices ( 
code TEXT, -- 銘柄 コード 
date TEXT, -- 日付 
open REAL, -- 初値 
high REAL, -- 高値 
low REAL, -- 安値 
close REAL, -- 終値
volume INTEGER, -- 出来高 
PRIMARY KEY( code, date) ); 

CREATE TABLE applied_divide_union_data ( 
code TEXT, -- 銘柄 コード 
date_of_right_allotment TEXT, -- 権利 確定 日 
PRIMARY KEY( code, date_of_right_allotment) 
);
