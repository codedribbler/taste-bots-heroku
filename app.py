from flask import Flask, render_template,request
from bokeh.embed import components
from bokeh.models import ColumnDataSource, HoverTool, PrintfTickFormatter
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from bokeh.palettes import magma,cividis,inferno,Set3
import pandas as pd
from bokeh.layouts import gridplot
from wordcloud import WordCloud
import base64
from io import BytesIO
import numpy as np
from PIL import Image
import pickle
from collections import Counter
from bokeh.plotting import figure
from bokeh.models import Label
from math import pi

app = Flask(__name__)


df= pd.read_csv('Pos_review_summary.csv')
df2 = pd.read_csv('Review_CD.csv')

app = Flask(__name__)

## Similar Restaurants

def Similar_Restaurants(name):
	n=9
	lst = list(df[df.label == df[df.restaurant==name]['label'].iloc[0]].sort_values(['review','rating'],ascending = False)['restaurant'][:n])
	#lst.remove(name)
	return lst

## Similar Restaurants page 2

def Similar_Restaurants2(name):
	n=9
	s=18
	lst = list(df[df.label == df[df.restaurant==name]['label'].iloc[0]].sort_values(['review','rating'],ascending = False)['restaurant'][n:s])
	#lst.remove(name)
	return lst


## Topic Recommendations
def Topic(Topic):
	n = 8
	lst = list(df.sort_values(['Service','rating'],ascending = False)['restaurant'][:8])
	return lst
	
## Topic Recommendations page 2
def Topic2(Topic):
    n = 8
    s = 16
    lst = list(df.sort_values([Topic,'rating'],ascending = False)['restaurant'][n:s])
    return lst
    


def style(p):
    p.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
    p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    p.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
    p.yaxis.minor_tick_line_color = None
    p.xgrid.grid_line_color = None

    p.xaxis.axis_line_width = 0.0

    p.yaxis.axis_line_width = 0.2
    p.yaxis.axis_line_color = "cadetblue"
    return p

def word_cloud(name):

    features={}
    with open('restaurant_word.pkl', 'rb') as handle:
        feature = pickle.load(handle)
    word_freq=dict(Counter(feature[name]))
    chef = np.array(Image.open('bam.png'))
    wc = WordCloud(width = 100, height = 100, background_color ='white',mask=chef, min_font_size = 5).generate_from_frequencies(word_freq)
    img = BytesIO()
    wc.to_image().save(img, 'PNG')
    img.seek(0)
    buffer = b''.join(img)
    b2 = base64.b64encode(buffer)
    wc_image=b2.decode('utf-8')
    return wc_image


def plot_top_five(frequent_feat):
    x=list(frequent_feat.Word)
    y=list(frequent_feat.Mean)
    print(f'x for hyderabadi biryani is {x}')
    print(f'y is {y}')
    p = figure(x_range=x,plot_height=400,toolbar_location=None,tools="")
    p = style(p)
    p.vbar(x=x,top=y,width=0.3,color=cividis(5))
    return p

def feature_sentiment_distribution(word,frequent_feat):
    x=list(frequent_feat.columns[-10:])

    y=list(frequent_feat[frequent_feat.Word == word][x].values[0])

    title = "Score Distribution for "+word
    p = figure(x_range=x, plot_height=300,plot_width=400, title=title,
           toolbar_location=None, tools="")

    p = style(p)
    p.vbar(x=x, top=y, width=0.3,color=inferno(10))
    return p

def create_donut(feature_df,ind,state):
    x=list(feature_df['Mean'])[ind]
    label = list(feature_df['Word'])[ind]
    end_angle= x/5 * pi


    my_pallete = ['#ff6e54','#ffa600','#58508d'] if state == 'high' else ['#B33A3A','#20639B','#F6D55C']
    p = figure(plot_height=250,plot_width=250, title=label, x_range=(-.5, .5),toolbar_location=None)

    p.annular_wedge(x=0, y=1,  inner_radius=0.17, outer_radius=0.24, direction="anticlock",
                start_angle=3.14, end_angle=end_angle,color=my_pallete[ind])
    mytext = Label(x=-0.07, y=.93, text=str(x),text_font_size = '15pt')
    p.add_layout(mytext)
    p = style(p)
    p.xgrid.visible = False
    p.ygrid.visible = False
    p.outline_line_color = None
    p.xaxis.major_label_text_font_size = '0pt'  # turn off x-axis tick labels
    p.yaxis.major_label_text_font_size = '0pt'  # turn off y-axis tick labels
    p.toolbar.logo = None
    p.toolbar_location = None
    return p


#Returning Total Reviews and Percentage

def Rating(restaurant):
    lst_rat = []
    lst_per = []
    df_test = round(df2[df2.restaurant==restaurant].rating.value_counts()).astype(int).rename_axis('rati').reset_index()
    df_test['Percentage'] = round((df_test.rating/sum(df_test.rating))*100)
    sum_rating = sum(df_test.rating)
    overall = round(sum(df_test.rati*df_test.rating)/sum(df_test.rating),1)

    for i in range(5):
        try:
            lst_per.append(df_test[df_test.rati == i+1].Percentage.iloc[0])
            lst_rat.append(df_test[df_test.rati == i+1].rating.iloc[0])


        except:
            print('Goes in except')
            lst_per.append(0)
            lst_rat.append(0)
    return sum_rating,overall,lst_per,lst_rat

def Star(rat_avg):
	if round(rat_avg) ==1:
		return ["fa fa-star checked","fa fa-star","fa fa-star","fa fa-star","fa fa-star",]
	elif round(rat_avg) ==2:
		return ["fa fa-star checked","fa fa-star checked","fa fa-star","fa fa-star","fa fa-star",]
	elif round(rat_avg) ==3:
		return ["fa fa-star checked","fa fa-star checked","fa fa-star checked","fa fa-star","fa fa-star",]
	elif round(rat_avg) ==4:
		return ["fa fa-star checked","fa fa-star checked","fa fa-star checked","fa fa-star checked","fa fa-star",]
	elif round(rat_avg) ==5:
		return ["fa fa-star checked","fa fa-star checked","fa fa-star checked","fa fa-star checked","fa fa-star checked"]

## Review Extraction
def Review(name):
    n = 3
    try:

        lst_tail = list(df2[df2.restaurant == name].sort_values('rating', ascending = True)['review'][:3])
        lst_top = list(df2[df2.restaurant == name].sort_values('rating', ascending = False)['review'][:3])
        return lst_tail,lst_top
    except:
        print('Please Provide Proper Restaurant Name')




@app.route('/')
def home():

    return render_template("index.html")
@app.route('/index2')
def index2():
    return render_template("index2.html")

@app.route('/Biryani')
def Biryani():
	Same = Topic(Topic = 'Biryani')
	return render_template("Biryani.html",Same = Same)

@app.route('/Biryani2')
def Biryani2():
	Same = Topic2(Topic = 'Biryani')
	return render_template("Biryani2.html",Same = Same)

@app.route('/Ambience')
def Ambience():
	Same = Topic(Topic = 'Ambience')
	return render_template("Ambience.html",Same = Same)

@app.route('/Ambience2')
def Ambience2():
	Same = Topic2(Topic = 'Ambience')
	return render_template("Ambience2.html",Same = Same)

@app.route('/Service')
def Service():
	Same = Topic(Topic = 'Service')
	
	return render_template("Service.html",Same = Same)

@app.route('/Service2')
def Service2():
	Same = Topic2(Topic = 'Service')
	return render_template("Service2.html",Same = Same)

@app.route('/Similar/<name1>')
def Similar(name1):
	Same = Similar_Restaurants(name = name1)
	return render_template("BBQ_Sim.html",restaurant_name = name1,Same = Same)

@app.route('/Similar2/<name2>')
def Similar2(name2):
	Same = Similar_Restaurants2(name = name2)
	return render_template("BBQ_Sim2.html",restaurant_name = name2,Same = Same)

@app.route('/search',methods=['GET', 'POST'])
def search():
	if request.method == 'POST':
		name = request.form['search']
		return barbecue(name)
#		name3 = name
#		if name == 'Barbeque Nation':
#			return render_template("barbeque_nation.html")
#		elif name == 'Absolute Barbecues':
#			return render_template("AB.html")
#		else:
#			return render_template("search.html")


@app.route('/restaurant_details/<name>')
def barbecue(name):
    print('The name is')
    print('The name is :',name)
    sentiment_df = pd.read_excel('sentiment_matrix.xlsx')
    sentiment_df = sentiment_df[sentiment_df.Restaurant == name]
    total_review,Avg_rating,percentages,rati = Rating(restaurant = name)
    Star_check = Star(rat_avg = Avg_rating)
    Neg_Rev,Pos_Rev = Review(name)
    if len(sentiment_df) != 0:

        frequent_feat=sentiment_df.sort_values(['Freq'],ascending=False).head(5)

        top_five = plot_top_five(frequent_feat)
        script_top_five,div_top_five = components(top_five)

        wc_image=word_cloud(name)

        top_five_word=list(frequent_feat['Word'].values)
        draw1 = feature_sentiment_distribution(top_five_word[0],frequent_feat)
        draw2 = feature_sentiment_distribution(top_five_word[1],frequent_feat)
        draw3 = feature_sentiment_distribution(top_five_word[2],frequent_feat)
        draw4 = feature_sentiment_distribution(top_five_word[3],frequent_feat)
        draw5 = feature_sentiment_distribution(top_five_word[4],frequent_feat)
        draw=gridplot([[draw1,draw2],[draw3,draw4],[None,draw5]])
        script_draw,div_draw = components(draw)


        top_feature=sentiment_df[(sentiment_df.Mean >=3.0) & (sentiment_df.Restaurant == name)].head(3)[['Word','Mean']]
        vals = top_feature.shape[0]
        plot_list=[create_donut(top_feature,val,'high') for val in range(0,vals)]
        high_rate=gridplot([plot_list])
        script_high_rate,div_high_rate = components(high_rate)


        bottom_feature=sentiment_df[(sentiment_df.Mean <=1.5) & (sentiment_df.Restaurant == name)].head(3)[['Word','Mean']]
        vals1 = bottom_feature.shape[0]
        plot_list=[create_donut(bottom_feature,val,'low') for val in range(0,vals1)]
        low_rate=gridplot([plot_list])
        script_low_rate,div_low_rate = components(low_rate)





        return render_template("restaurant_detail.html",restaurant_name=name,wc_image=wc_image,script_top_five=script_top_five,div_top_five=div_top_five,script_draw=script_draw,div_draw=div_draw,script_high_rate=script_high_rate,div_high_rate=div_high_rate,script_low_rate=script_low_rate,div_low_rate=div_low_rate,total_review = total_review,Avg_rating = Avg_rating,percentages = percentages,rati=rati,Star_check=Star_check,Neg_Rev=Neg_Rev,Pos_Rev=Pos_Rev)

    else:
    	return render_template("oops.html",restaurant_name=name)


@app.route('/coming_soon')
def coming_soon():
	return render_template("CS.html")




if __name__ == '__main__':
	app.run()
