'''

'''
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from collections import Counter
from PyPDF2 import PdfFileMerger

class SalesAnalysis:

    def __init__(self,df1,df,pdfs):
        '''

        '''
        self.df1= df1
        self.df=df
        self.pdfs=pdfs

    def city(address):
        '''

        '''
        return address.split(",")[1].strip(" ")

    def state(address):
        '''

        '''
        return address.split(",")[2].split(" ")[1]

    def mergecleandata():
        '''

        '''
        try:

            extension = 'csv'
            all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
            mData = pd.concat([pd.read_csv(f) for f in all_filenames])
            mData = mData.dropna(how='any', axis=0)
            mData.drop(mData.loc[mData['Order ID'] == 'Order ID'].index, inplace=True)
            mData.drop(mData.loc[mData['Product'] == 'Product'].index, inplace=True)
            mData.drop(mData.loc[mData['Quantity Ordered'] == 'Quantity Ordered'].index, inplace=True)
            mData.drop(mData.loc[mData['Price Each'] == 'Price Each'].index, inplace=True)
            mData.drop(mData.loc[mData['Order Date'] == 'Order Date'].index, inplace=True)
            mData.drop(mData.loc[mData['Purchase Address'] == 'Purchase Address'].index, inplace=True)
            mData.to_csv("Sales_2019_Merged.csv", index=False, encoding='utf-8-sig')

        except:

            print('Function Error: mergecleandata')

    def additionalfeatures(self):
        '''

        '''
        try:

            self.df = self.df[self.df['Order Date'].str[0:2] != 'Or']
            self.df['Quantity Ordered'] = pd.to_numeric(self.df['Quantity Ordered'])
            self.df['Price Each'] = pd.to_numeric(self.df['Price Each'])
            self.df['Month 2'] = pd.to_datetime(self.df['Order Date']).dt.month
            self.df['Month'] = self.df['Order Date'].str[0:2]
            self.df['Month'] = self.df['Month'].astype('int32')
            self.df['City'] = self.df['Purchase Address'].apply(lambda x: f"{city(x)}  ({state(x)})")
            self.df['Sales'] = self.df['Quantity Ordered'].astype('int') * self.df['Price Each'].astype('float')
            self.df.groupby(['Month']).sum()
            self.df['Hour'] = pd.to_datetime(df['Order Date']).dt.hour
            self.df['Minute'] = pd.to_datetime(df['Order Date']).dt.minute
            self.df['Count'] = 1
            self.df.to_excel('Sales_2019_Updated.xlsx')

        except:

            print('Function Error: additionalfeatures')

    def datavisualization1(self):
        '''

        '''
        try:

            months = [1,2,3,4,5,6,7,8,9,10,11,12]
            sns.barplot(x=months,y= self.df1.groupby(['Month']).sum()['Sales'],palette='coolwarm')
            plt.xticks(months)
            plt.title('Month over Month Sales Data')
            plt.ylabel('Sales($)')
            plt.xlabel('Month')
            plt.grid(True)
            plt.savefig('Month over month sales data.pdf')

        except:

            print('Function Error: datavisualization1')

    def datavisualization2(self):
        '''

        '''
        try:

            key1 = [city for city, df in self.df1.groupby(['City'])]
            np.random.seed(19680801)
            plt.rcdefaults()
            fig, ax1 = plt.subplots()
            y_pos = np.arange(len(key1))
            performance = self.df1.groupby(['City']).sum()['Sales']
            error = np.random.rand(len(key1))
            ax1.barh(y_pos, performance, xerr=error, align='center')
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(key1)
            plt.xlabel('Sales in USD ($)')
            plt.ylabel('City')
            plt.title('City wise Sales Data')
            ax1.invert_yaxis()
            plt.grid(True)
            plt.savefig('City wise sales data.pdf')

        except:

            print('Function Error: datavisualization2')

    def datavisualization3(self):
        '''

        '''
        try:

            key2 = [pair1 for pair1, df1 in self.df1.groupby(['Hour'])]
            plt.plot(key2, self.df1.groupby(['Hour']).count()['Count'])
            plt.xticks(key2)
            plt.ylabel('Sales')
            plt.xlabel('Hours of Day')
            plt.title('Most Busy Times for sales')
            plt.grid(True)
            plt.savefig('Most Busy Times for sales.pdf')

        except:

            print('Function Error: datavisualization3')

    def datavisualization4(self):
        '''

        '''
        try:

            df_g= self.df1[self.df1['Order ID'].duplicated(keep=False)]
            df_g['Grouped'] = df_g.groupby('Order ID')['Product'].transform(lambda x: ','.join(x))
            df2 = df_g[['Order ID', 'Grouped']].drop_duplicates()
            count = Counter()

            for row in df2['Grouped']:
                row_list = row.split(',')
                count.update(Counter(combinations(row_list, 2)))

            for key, value in count.most_common(10):
                print(key, value)

            product_group = self.df1.groupby('Product')
            quantity_ordered = product_group.sum()['Quantity Ordered']
            key4 = [pair for pair, dft in product_group]
            plt.bar(key4, quantity_ordered)
            plt.title('Quantity Trend')
            plt.grid(True)
            plt.xticks(key4, rotation='vertical', size=8)
            count = Counter()

            for row in df2['Grouped']:
                row_list = row.split(',')
                count.update(Counter(combinations(row_list, 2)))

            for k,value in count.most_common(10):
                print(k, value)

            product_group = self.df1.groupby('Product')
            quantity_ordered = product_group.sum()['Quantity Ordered']
            key3 = [pair for pair, df_g in product_group]
            plt.bar(key3, quantity_ordered)
            plt.xticks(key3, rotation=20, size=4)
            plt.savefig('Quantity v product.pdf')

        except:

            print('Function Error: datavisualization4')

    def datavisualization5(self):
        '''
        This function will
        '''
        try:

            df_g1 = self.df1[self.df1['Order ID'].duplicated(keep=False)]
            df_g1['Grouped'] = df_g1.groupby('Order ID')['Product'].transform(lambda x: ','.join(x))
            df_2 = df_g1[['Order ID', 'Grouped']].drop_duplicates()

            count = Counter()

            for row in df_2['Grouped']:
                row_list = row.split(',')
                count.update(Counter(combinations(row_list, 2)))

            for k2, value1 in count.most_common(10):
                print(k2, value1)

            product_group = self.df1.groupby('Product')
            quantity_ordered = product_group.sum()['Quantity Ordered']
            key5 = [pairx for pairx, dft1 in product_group]
            df_g1 = self.df1[self.df1['Order ID'].duplicated(keep=False)]
            df_g1['Grouped'] = df_g1.groupby('Order ID')['Product'].transform(lambda x: ','.join(x))
            prices1 = self.df1.groupby('Product').mean()['Price Each']
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            ax1.bar(key5, quantity_ordered, color='b')
            ax2.plot(key5, prices1, color='r')
            ax1.set_xlabel('Product Name')
            ax1.set_ylabel('Quantity Ordered', color='b')
            ax2.set_ylabel('Price ($)', color='r')
            ax1.set_xticklabels(key5, rotation=30, size=4)
            plt.title('Quantity/Price analysis per product')
            plt.grid(True)
            plt.savefig('Test.pdf')

        except:

            print('Function Error: datavisualization5')

    def pdfMerge(self):
        '''
        This function will merge all the output pdfs into one pdf
        '''
        try:

            merger = PdfFileMerger()
            for pdf in self.pdfs:
                merger.append(pdf)
            merger.close()
            merger.write("SalesReport.pdf")
        except:
            print('Function Error: pdfMerge')

if __name__ == '__main__':
    _salesAnalysis_= SalesAnalysis(df=pd.read_csv('Sales_2019_Merged.csv'),
                                   df1=pd.read_excel('Sales_2019_Updated.xlsx')
                                   ,pdfs= ['Month over month sales data.pdf', 'City wise sales data.pdf', 'QuantityProduct.pdf', 'Price Quantity .pdf','Most Busy Times for sales.pdf'])
    _salesAnalysis_.pdfMerge()
    # _salesAnalysis_.datavisualization1()
    # _salesAnalysis_.datavisualization2()
    # _salesAnalysis_.datavisualization3()
    # _salesAnalysis_.datavisualization4()
    # _salesAnalysis_.datavisualization5()
    # _salesAnalysis_.additionalfeatures()

