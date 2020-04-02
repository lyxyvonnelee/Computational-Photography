#include "GrabCut.h"
#include "stdio.h"
#include "graph.h"
#include "gcgraph.hpp"


using namespace cv;
using namespace std;

GrabCut2D::~GrabCut2D(void)
{
}

//init GMM
class GMM {
public:
	static const int componentsCount = 5; //K=5
	GMM(Mat& _model);
	void endLearning();
	void initLearning();
	void addSample(int ci, const Vec3d color);
	int whichComponent(const Vec3d color)const;
	double operator()(const Vec3d color) const;
	double operator()(int ci, const Vec3d color)const;

private:
	Mat model;
	double* coefs;
	double* mean;
	double* cov;
	void calcInverseCovAndDeterm(int ci);  //calculate cov inverse and determinant
	double inverseCovs[componentsCount][3][3];
	double covDeterms[componentsCount];
	double sums[componentsCount][3];
	double prods[componentsCount][3][3];
	int sampleCounts[componentsCount];
	int totalSampleCount;
	
};

void GMM::calcInverseCovAndDeterm(int ci)
{
	if (coefs[ci]>0){

		double *c = cov + 9 * ci;
		double dtrm =
			covDeterms[ci] = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) + c[2] * (c[3] * c[7] - c[4] * c[6]);

		CV_Assert(dtrm > std::numeric_limits<double>::epsilon());

		inverseCovs[ci][0][0] = (c[4] * c[8] - c[5] * c[7]) / dtrm;
		inverseCovs[ci][1][0] = -(c[3] * c[8] - c[5] * c[6]) / dtrm;
		inverseCovs[ci][2][0] = (c[3] * c[7] - c[4] * c[6]) / dtrm;
		inverseCovs[ci][0][1] = -(c[1] * c[8] - c[2] * c[7]) / dtrm;
		inverseCovs[ci][1][1] = (c[0] * c[8] - c[2] * c[6]) / dtrm;
		inverseCovs[ci][2][1] = -(c[0] * c[7] - c[1] * c[6]) / dtrm;
		inverseCovs[ci][0][2] = (c[1] * c[5] - c[2] * c[4]) / dtrm;
		inverseCovs[ci][1][2] = -(c[0] * c[5] - c[2] * c[3]) / dtrm;
		inverseCovs[ci][2][2] = (c[0] * c[4] - c[1] * c[3]) / dtrm;
	}
}


GMM::GMM(Mat& _model)
{
	
	const int modelSize = 3+9+1;   //mean, covariance, component weight  
		_model.create(1, modelSize*componentsCount, CV_64FC1 );
		_model.setTo(Scalar(0));

	model = _model;
	
	coefs = model.ptr<double>(0);
	mean  = coefs  + componentsCount;
	cov = mean + 3 * componentsCount;

	for (int ci = 0; ci < componentsCount; ci++)
		if (coefs[ci] > 0)
			calcInverseCovAndDeterm(ci);

}

void GMM::endLearning()
{
	const double variance = 0.01;
	for (int ci = 0; ci < componentsCount; ci++) {
		int n = sampleCounts[ci];
		if (n == 0)
			coefs[ci] = 0;
		else
		{
			coefs[ci] = (double)n / totalSampleCount;
		}
		
		double* m = mean + 3 * ci;
		m[0] = sums[ci][0] / n;
		m[1] = sums[ci][1] / n;
		m[2] = sums[ci][2] / n;


		double* c = cov + 9 * ci;
		c[0] = prods[ci][0][0] / n - m[0] * m[0]; 
		c[1] = prods[ci][0][1] / n - m[0] * m[1]; 
		c[2] = prods[ci][0][2] / n  - m[0] * m[2];
		c[3] = prods[ci][1][0] / n - m[1] * m[0]; 
		c[4] = prods[ci][1][1] / n - m[1] * m[1]; 
		c[5] = prods[ci][1][2] / n  - m[1] * m[2];
		c[6] = prods[ci][2][0] / n - m[2] * m[0]; 
		c[7] = prods[ci][2][1] / n - m[2] * m[1]; c[8] = prods[ci][2][2] / n  - m[2] * m[2];


		double dtrm = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) + c[2] * (c[3] * c[7] - c[4] * c[6]);
		if (dtrm  <= std::numeric_limits<double>::epsilon())
		{

			c[0] += variance;
			c[4] += variance;
			c[8] += variance;
		}

		calcInverseCovAndDeterm(ci);
		
	}
	
}

void GMM::addSample(int ci, const Vec3d color)
{
	sums[ci][0] += color[0]; sums[ci][1] += color[1]; sums[ci][2] += color[2];
	prods[ci][0][0] += color[0] * color[0]; prods[ci][0][1] += color[0] * color[1]; prods[ci][0][2] += color[0] * color[2];
	prods[ci][1][0] += color[1] * color[0]; prods[ci][1][1] += color[1] * color[1]; prods[ci][1][2] += color[1] * color[2];
	prods[ci][2][0] += color[2] * color[0]; prods[ci][2][1] += color[2] * color[1]; prods[ci][2][2] += color[2] * color[2];
	sampleCounts[ci]++;
	totalSampleCount++;
}

double GMM::operator()(const Vec3d color)const
{
	double res = 0;
	for (int ci = 0; ci< componentsCount; ci++)
		res += coefs[ci] * (*this)(ci, color);
	return res;
}


double GMM::operator()(int ci, const Vec3d color) const
{
	double res  = 0;
	if (coefs[ci] > 0)
	{     
		Vec3d diff  = color;
		double* m  = mean + 3 * ci;
		diff[0] -= m[0]; diff[1] -= m[1]; diff[2] -= m[2];
		double mult = diff[0] * (diff[0] * inverseCovs[ci][0][0] + diff[1] * inverseCovs[ci][1][0] + diff[2] * inverseCovs[ci][2][0])
			+ diff[1] * (diff[0] * inverseCovs[ci][0][1] + diff[1] * inverseCovs[ci][1][1] + diff[2] * inverseCovs[ci][2][1])
			+ diff[2] * (diff[0] * inverseCovs[ci][0][2] + diff[1] * inverseCovs[ci][1][2] + diff[2] * inverseCovs[ci][2][2]);
		res = 1.0f / sqrt(covDeterms[ci])* exp(-0.5f*mult);
	}
	return res;
}

int GMM::whichComponent(const Vec3d color)const
{
	int k = 0;
	double max = 0;

	for(int ci = 0; ci < componentsCount; ci++)
	{
		double p = (*this)(ci, color);
		if (p>max)
		{
			k = ci;
			max = p;
		}
	}
	return k;
}

static void constructGCGraph(const Mat& img, const Mat& mask, const GMM& bgdGMM, const GMM& fgdGMM, double lambda,
	const Mat& leftW, const Mat& upleftW, const Mat& upW, const Mat& uprightW,GCGraph<double>& graph )
{
	int vtxCount = img.cols*img.rows;
	int edgeCount = 2 * (4 * vtxCount - 3 * (img.cols + img.rows) + 2);
																		
	graph.create(vtxCount, edgeCount);
	Point p;
	for(p.y = 0; p.y< img.rows; p.y++)
	{
		for (p.x = 0; p.x< img.cols; p.x++)
		{
			// add node  
			int vtxIdx = graph.addVtx();
			Vec3b color = img.at<Vec3b>(p);

			// set t-weights              
			double fromSource, toSink;
			if (mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD)
			{

				fromSource = -log(bgdGMM(color));
				toSink = -log(fgdGMM(color));
			}
			else if(mask.at<uchar>(p) == GC_BGD)
			{
				
				fromSource = 0;
				toSink = lambda;
			}
			else{
				fromSource = lambda;
			    toSink = 0;
			}
			
			graph.addTermWeights(vtxIdx, fromSource, toSink);

			// set n-weights  n-links    
			if (p.x>0)
			{
				double w= leftW.at<double>(p);
				graph.addEdges(vtxIdx, vtxIdx - 1,w, w);
			}
			if (p.x>0 && p.y>0)
			{
				double w = upleftW.at<double>(p);
				graph.addEdges(vtxIdx, vtxIdx - img.cols - 1, w, w);
			}
			if (p.y>0)
			{
				double w = upW.at<double>(p);
				graph.addEdges(vtxIdx, vtxIdx - img.cols,w, w);
			}
			if (p.x<img.cols - 1 && p.y>0)
			{
				double w = uprightW.at<double>(p);
				graph.addEdges(vtxIdx, vtxIdx - img.cols + 1,w, w);
			}
		}
	}
}


void GMM::initLearning()
{
	for(int ci = 0; ci < componentsCount; ci++)
	{
		sums[ci][0] = sums[ci][1] = sums[ci][2] = 0;
		prods[ci][0][0] = prods[ci][0][1] = prods[ci][0][2] = 0;
		prods[ci][1][0] = prods[ci][1][1] = prods[ci][1][2] = 0;
		prods[ci][2][0] = prods[ci][2][1] = prods[ci][2][2] = 0;
		sampleCounts[ci] = 0;
		//cout << prods[ci];
		
	}
	totalSampleCount = 0;
}






void GrabCut2D::GrabCut( cv::InputArray _img, cv::InputOutputArray _mask, cv::Rect rect, cv::InputOutputArray _bgdModel,cv::InputOutputArray _fgdModel, int iterCount, int mode )
{
   // std::cout<<"Execute GrabCut Function: Please finish the code here!"<<std::endl;


//一.参数解释：
	//输入：
	 //cv::InputArray _img,     :输入的color图像(类型-cv:Mat)
     //cv::Rect rect            :在图像上画的矩形框（类型-cv:Rect) 
  	//int iterCount :           :每次分割的迭代次数（类型-int)


	//中间变量
	//cv::InputOutputArray _bgdModel ：   背景模型（推荐GMM)（类型-13*n（组件个数）个double类型的自定义数据结构，可以为cv:Mat，或者Vector/List/数组等）
	//cv::InputOutputArray _fgdModel :    前景模型（推荐GMM) （类型-13*n（组件个数）个double类型的自定义数据结构，可以为cv:Mat，或者Vector/List/数组等）


	//输出:
	//cv::InputOutputArray _mask  : 输出的分割结果 (类型： cv::Mat)

//二. 伪代码流程：
	//1.Load Input Image: 加载输入颜色图像;
	//2.Init Mask: 用矩形框初始化Mask的Label值（确定背景：0， 确定前景：1，可能背景：2，可能前景：3）,矩形框以外设置为确定背景，矩形框以内设置为可能前景;
	//3.Init GMM: 定义并初始化GMM(其他模型完成分割也可得到基本分数，GMM完成会加分）
	//4.Sample Points:前背景颜色采样并进行聚类（建议用kmeans，其他聚类方法也可)
	//5.Learn GMM(根据聚类的样本更新每个GMM组件中的均值、协方差等参数）
	//4.Construct Graph（计算t-weight(数据项）和n-weight（平滑项））
	//7.Estimate Segmentation(调用maxFlow库进行分割)
	//8.Save Result输入结果（将结果mask输出，将mask中前景区域对应的彩色图像保存和显示在交互界面中）



	//load input image
	Mat img = _img.getMat();
	Mat &bgdModel = _bgdModel.getMatRef();
	Mat &fgdModel = _bgdModel.getMatRef();

	GMM bgdGMM(bgdModel), fgdGMM(fgdModel);
	Mat compIdxs(img.size(), CV_32SC1);
	Mat mask = _mask.getMat();   //Mat::zeros(img.rows,img.cols,CV_32SC1);

	
	
		if (mode == GC_INIT_WITH_RECT) {
			//init mask		                         
			Point2i tl, br;
			tl = rect.tl();
			br = rect.br();
			for (int i = 0; i < mask.rows; ++i) {
				for (int j = 0; j < mask.cols; ++j) {
					if (i<max(tl.y, br.y) && i>min(tl.y, br.y) && j > tl.x&&j < br.x) {
						mask.at<int>(i, j) = 3;
					}                                                                         //inside the rect set to posssible foreground
					else {
						mask.at<int>(i, j) = 0;                                               //outside the rect set to background
					}

				}
			}
		}


	
	//Init GMM
		const int kMeansItCount = 10;
		const int kMeansType = KMEANS_PP_CENTERS;
		Mat bgdLabels, fgdLabels;
		vector<Vec3f> bgdSamples, fgdSamples;

		for (int i = 0; i < img.rows; i++)
		{
			for (int j = 0; j< img.cols; j++)
			{

				if (mask.at<uchar>(i, j) == 0 || mask.at<uchar>(i, j) == 2)
					bgdSamples.push_back((Vec3f)img.at<Vec3b>(i, j));
				else                                              // GC_FGD | GC_PR_FGD  
					fgdSamples.push_back((Vec3f)img.at<Vec3b>(i, j));
			}
		}


		Mat bgdsamples((int)bgdSamples.size(), 3, CV_32FC1, &bgdSamples[0][0]);
		Mat fgdsamples((int)fgdSamples.size(), 3, CV_32FC1, &fgdSamples[0][0]);

		kmeans(bgdsamples, GMM::componentsCount, bgdLabels,
			TermCriteria(CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType);
		kmeans(fgdsamples, GMM::componentsCount, fgdLabels,
			TermCriteria(CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType);

		bgdGMM.initLearning();
		for (int i = 0; i < (int)bgdSamples.size(); i++)
			bgdGMM.addSample(bgdLabels.at<int>(i, 0), bgdSamples[i]);
		bgdGMM.endLearning();

		fgdGMM.initLearning();
		for (int i = 0; i < (int)fgdSamples.size(); i++)
			fgdGMM.addSample(fgdLabels.at<int>(i, 0), fgdSamples[i]);
		fgdGMM.endLearning();
	

	//show results of kmeans
	//Mat ImgRes(img.size(), CV_8UC3);
	//Mat_<Vec3b>::iterator itRes = ImgRes.begin<Vec3b>();
	//Mat_<int>::iterator itLabel = fgdlabels.begin<int>();
	//for (; itLabel  != fgdlabels.end<int>(); ++itLabel, ++itRes)
	//	*itRes = fgdSamples.at<Vec3f>(*itLabel, 0);

	////Mat ImgRes1(img.size(), CV_8UC3);
	//Mat_<Vec3b>::iterator itRes1 = ImgRes.begin<Vec3b>();
	//Mat_<int>::iterator itLabel1 = bgdlabels.begin<int>();
	//for (; itLabel1 != bgdlabels.end<int>(); ++itLabel1, ++itRes1)
	//	*itRes1 = bgdSamples.at<Vec3f>(*itLabel1, 0);

	//cvtColor(ImgRes, ImgRes, CV_HSV2BGR);
	//namedWindow("KMEANS");
	//namedWindow("img");
	//imshow("KMEANS", ImgRes);
	//imshow("img", img);
	//waitKey(0);

	if (iterCount <= 0)
		return;



	//calculate beta
	const double gamma = 50;
	const double lambda = 9 * gamma;	
	double beta = 0;

	for (int y = 0; y < img.rows; y++)
	{
		for (int x = 0; x < img.cols; x++)
		{
			Vec3d color = img.at<Vec3b>(y, x);
			if (x>0) // left 
			{
				Vec3d diff = color - (Vec3d)img.at<Vec3b>(y, x - 1);
				beta += diff.dot(diff);
			}
			if (y>0 && x>0) // upleft 
			{
				Vec3d diff = color - (Vec3d)img.at<Vec3b>(y - 1, x - 1);
				beta += diff.dot(diff);
			}
			if (y>0) // up 
			{
				Vec3d diff = color - (Vec3d)img.at<Vec3b>(y - 1, x);
				beta += diff.dot(diff);
			}
			if (y>0 && x<img.cols - 1) // upright 
			{
				Vec3d diff = color - (Vec3d)img.at<Vec3b>(y - 1, x + 1);
				beta += diff.dot(diff);
			}
		}
	}
	if (beta <= std::numeric_limits<double>::epsilon())
		beta = 0;
	else 
		beta = 1.f / (2 * beta / (4 * img.cols*img.rows - 3 * img.cols - 3 * img.rows + 2));

	const double b = beta;


	//calculate Nweights
	Mat leftW, upleftW, upW, uprightW;
    const double gammaDiv = gamma / std::sqrt(2.0f);

		leftW.create(img.rows, img.cols, CV_64FC1);
		upleftW.create(img.rows, img.cols, CV_64FC1);
		upW.create(img.rows, img.cols, CV_64FC1);
		uprightW.create(img.rows, img.cols, CV_64FC1);
		for (int y = 0; y < img.rows; y++)
		{
			for (int x = 0; x < img.cols; x++)
			{
				Vec3d color = img.at<Vec3b>(y, x);
				if (x - 1 >= 0)
				{
					Vec3d diff = color - (Vec3d)img.at<Vec3b>(y, x - 1);
					leftW.at<double>(y, x) = gamma * exp(-beta*diff.dot(diff));
				}
				else
					leftW.at<double>(y, x) = 0;
				if (x - 1 >= 0 && y - 1 >= 0)
				{
					Vec3d diff = color - (Vec3d)img.at<Vec3b>(y - 1, x - 1);
					upleftW.at<double>(y, x) = gammaDiv * exp(-beta*diff.dot(diff));
				}
				else
					upleftW.at<double>(y, x) = 0;
				if (y - 1 >= 0)
				{
					Vec3d diff = color - (Vec3d)img.at<Vec3b>(y - 1, x);
					upW.at<double>(y, x) = gamma * exp(-beta*diff.dot(diff));
				}
				else
					upW.at<double>(y, x) = 0;
				if (x + 1 < img.cols && y - 1 >= 0)
				{
					Vec3d diff = color - (Vec3d)img.at<Vec3b>(y - 1, x + 1);
					uprightW.at<double>(y, x) = gammaDiv * exp(-beta*diff.dot(diff));
				}
				else
					uprightW.at<double>(y, x) = 0;
			}
		}

		for (int i = 0; i < iterCount; i++)
		{
			GCGraph<double> graph;
	
			for(int i = 0; i < img.rows; i++)
			{
				for (int j= 0;j < img.cols; j++)
				{
				    Vec3d color = img.at<Vec3b>(i,j);
					compIdxs.at<int>(i,j) = mask.at<uchar>(i,j) == 0  || mask.at<uchar>(i,j) == 2 ?
						bgdGMM.whichComponent(color) : fgdGMM.whichComponent(color);
				}
			}

			

			
			for(int ci = 0; ci < GMM::componentsCount; ci++)
			{
				for(int i = 0; i < img.rows;i++)
				{
					for(int j = 0; j < img.cols; j++)
					{
						if (compIdxs.at<int>(i,j) == ci)
						{
							if (mask.at<uchar>(i,j) == 0 || mask.at<uchar>(i,j) == 2)
								bgdGMM.addSample(ci, img.at<Vec3b>(i,j));
							else
								fgdGMM.addSample(ci, img.at<Vec3b>(i,j));
						}
					}
				}
			}

			bgdGMM.endLearning();
			fgdGMM.endLearning();

			constructGCGraph(img, mask, bgdGMM, fgdGMM, lambda, leftW, upleftW, upW, uprightW, graph);
		
				graph.maxFlow();
				for (int i = 0; i < mask.rows; i++)
				{
					for (int j = 0; j < mask.cols; j++)
					{

						if (mask.at<uchar>(i,j) == 2 || mask.at<uchar>(i,j) == 3)
						{
							if (graph.inSourceSegment(i*mask.cols + j))
								mask.at<uchar>(i,j) = 3;
							else
								mask.at<uchar>(i,j) = 2;
						}
					}
				}
		}
		
}