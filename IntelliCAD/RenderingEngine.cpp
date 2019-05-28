/*
*	Copyright (C) 2019 Jin Won. All right reserved.
*
*	파일명			: RenderingEngine.cpp
*	작성자			: 원진
*	최종 수정일		: 19.04.07
*/

#include "custom_cudart.h"
#include "RenderingEngine.h"
#include "SystemIndirectAccessor.h"
#include "RayBoxInstersector.h"
#include "NumberUtility.hpp"
#include "Constant.h"
#include <limits>
#include "Index2D.hpp"

RenderingEngine RenderingEngine::__instance;

using namespace std;

namespace Device
{
	namespace Volume
	{
		/* variable */
		/// <summary>
		/// <para>GPU 메모리를 할당하는 볼륨 메타 데이터</para>
		/// <para>볼륨 크기, 복셀간 간격, 복셀 정밀도를 가진다.</para>
		/// </summary>
		__device__
		static VolumeMeta __meta[1];

		/// <summary>
		/// <para>GPU 메모리를 할당하는 short 타입 볼륨 3차원 배열</para>
		/// <para>데이터를 0.f~1.f 사이로 정규화해서 저장한다.</para>
		/// </summary>
		static cudaArray_t __pBuffer = nullptr;

		/// <summary>
		/// <para>GPU 볼륨 3차원 배열과 바인딩 된 텍스처 인터페이스</para>
		/// <para>float 타입 3차원 index로 접근하여 대응되는 볼륨 데이터를 선형 보간해서 가져온다.</para>
		/// <para>접근 예시: tex3D(texMem, x, y, z);</para>
		/// </summary>
		static texture<ushort, 3, cudaReadModeNormalizedFloat> __texMem;

		/* function */
		/// <summary>
		/// GPU 메모리를 할당하고, 텍스처 인터페이스와 바인딩 한다.
		/// </summary>
		/// <param name="volumeData">
		/// <para>CPU 볼륨 데이터 레퍼런스</para>
		/// <para>메타 데이터를 포함한다.</para>
		/// </param>
		static void __malloc(const VolumeData& volumeData)
		{
			const cudaChannelFormatDesc DESC = cudaCreateChannelDesc<ushort>();

			// 볼륨 메타 데이터를 CPU에서 GPU 변수로 복사한다.  
			cudaMemcpyToSymbol(__meta, &volumeData.meta, sizeof(VolumeMeta));

			// 볼륨의 크기
			const cudaExtent VOL_SIZE =
				make_cudaExtent(volumeData.meta.size.width, volumeData.meta.size.height, volumeData.meta.size.depth);

			// 볼륨 3차원 배열 메모리를 GPU에 할당한다.
			cudaMalloc3DArray(&__pBuffer, &DESC, VOL_SIZE);

			// 메모리 복사 파라미터를 설정한다.
			cudaMemcpy3DParms params;
			ZeroMemory(&params, sizeof(cudaMemcpy3DParms));

			params.srcPtr = make_cudaPitchedPtr(
				volumeData.pBuffer.get(), (VOL_SIZE.width * sizeof(ushort)), VOL_SIZE.width, VOL_SIZE.height);
			
			params.dstArray = __pBuffer;
			params.extent = VOL_SIZE;

			// 볼륨 데이터를 CPU에서 GPU 배열로 복사한다.
			cudaMemcpy3D(&params);

			__texMem.addressMode[0] = cudaAddressModeBorder;
			__texMem.addressMode[1] = cudaAddressModeBorder;
			__texMem.addressMode[2] = cudaAddressModeBorder;
			__texMem.filterMode = cudaFilterModeLinear;

			// 텍스처 참조자와 볼륨 3차원 배열을 바인딩한다.
			cudaBindTextureToArray(__texMem, __pBuffer);
		}

		/// <summary>
		/// GPU 메모리를 반환하고, 텍스처 인터페이스와 바인딩을 해제한다.
		/// </summary>
		static void __free()
		{
			if (__pBuffer)
			{
				cudaUnbindTexture(__texMem);
				cudaFreeArray(__pBuffer);

				__pBuffer = nullptr;
			}
		}
	}

	namespace TransferFunc
	{
		/* variable */
		/// <summary>
		/// <para>GPU 메모리를 할당하는 float 타입 1차원 red transfer function</para>
		/// <para>0.f~1.f 사이 값을 가진다.</para>
		/// </summary>
		static cudaArray_t __pRed = nullptr;

		/// <summary>
		/// <para>GPU 메모리를 할당하는 float 타입 1차원 green transfer function</para>
		/// <para>0.f~1.f 사이 값을 가진다.</para>
		/// </summary>
		static cudaArray_t __pGreen = nullptr;

		/// <summary>
		/// <para>GPU 메모리를 할당하는 float 타입 1차원 blue transfer function</para>
		/// <para>0.f~1.f 사이 값을 가진다.</para>
		/// </summary>
		static cudaArray_t __pBlue = nullptr;

		/// <summary>
		/// <para>GPU 메모리를 할당하는 float 타입 1차원 alpha transfer function</para>
		/// <para>0.f~1.f 사이 값을 가진다.</para>
		/// </summary>
		static cudaArray_t __pAlpha = nullptr;

		/// <summary>
		/// <para>GPU red 필터와 바인딩 된 텍스처 접근자</para>
		/// <para>0.f~1.f 사이로 정규화 된 float 타입 1차원 index로 접근할 수 있다.</para>
		/// <para>이때 대응되는 값은 red 필터를 선형 보간하여 가져온다.</para>
		/// </summary>
		static texture<float> __texMemRed;

		/// <summary>
		/// <para>GPU green 필터와 바인딩 된 텍스처 접근자</para>
		/// <para>0.f~1.f 사이로 정규화 된 float 타입 1차원 index로 접근할 수 있다.</para>
		/// <para>이때 대응되는 값은 green 필터를 선형 보간하여 가져온다.</para>
		/// </summary>
		static texture<float> __texMemGreen;

		/// <summary>
		/// <para>GPU blue 필터와 바인딩 된 텍스처 접근자</para>
		/// <para>0.f~1.f 사이로 정규화 된 float 타입 1차원 index로 접근할 수 있다.</para>
		/// <para>이때 대응되는 값은 blue 필터를 선형 보간하여 가져온다.</para>
		/// </summary>
		static texture<float> __texMemBlue;

		/// <summary>
		/// <para>GPU alpha 필터와 바인딩 된 텍스처 접근자</para>
		/// <para>0.f~1.f 사이로 정규화 된 float 타입 1차원 index로 접근할 수 있다.</para>
		/// <para>이때 대응되는 값은 alpha 필터를 선형 보간하여 가져온다.</para>
		/// </summary>
		static texture<float> __texMemAlpha;

		/* function */
		/// <summary>
		/// transfer function을 CPU에서 GPU로 복사한다. 
		/// </summary>
		/// <param name="transferFunc">
		/// CPU trasfer function 레퍼런스
		/// </param>
		static void __memcpy(const TransferFunction& transferFunc)
		{
			const size_t MEM_SIZE = (transferFunc.PRECISION * sizeof(float));

			cudaMemcpyToArray(__pRed, 0, 0, transferFunc.getRed(), MEM_SIZE, cudaMemcpyHostToDevice);
			cudaMemcpyToArray(__pGreen, 0, 0, transferFunc.getGreen(), MEM_SIZE, cudaMemcpyHostToDevice);
			cudaMemcpyToArray(__pBlue, 0, 0, transferFunc.getBlue(), MEM_SIZE, cudaMemcpyHostToDevice);
			cudaMemcpyToArray(__pAlpha, 0, 0, transferFunc.getAlpha(), MEM_SIZE, cudaMemcpyHostToDevice);
		}

		/// <summary>
		/// GPU 메모리를 할당하고, 텍스처 인터페이스와 바인딩 한다.
		/// </summary>
		/// <param name="transferFunc">
		/// CPU trasfer function 레퍼런스
		/// </param>
		static void __malloc(const TransferFunction& transferFunc)
		{
			const cudaChannelFormatDesc DESC = cudaCreateChannelDesc<float>();
			const size_t MEM_SIZE = (transferFunc.PRECISION * sizeof(float));

			cudaMallocArray(&__pRed, &DESC, MEM_SIZE);
			cudaMallocArray(&__pGreen, &DESC, MEM_SIZE);
			cudaMallocArray(&__pBlue, &DESC, MEM_SIZE);
			cudaMallocArray(&__pAlpha, &DESC, MEM_SIZE);

			__memcpy(transferFunc);

			__texMemRed.filterMode = cudaFilterModeLinear;
			cudaBindTextureToArray(__texMemRed, __pRed);

			__texMemGreen.filterMode = cudaFilterModeLinear;
			cudaBindTextureToArray(__texMemGreen, __pGreen);

			__texMemBlue.filterMode = cudaFilterModeLinear;
			cudaBindTextureToArray(__texMemBlue, __pBlue);

			__texMemAlpha.filterMode = cudaFilterModeLinear;
			cudaBindTextureToArray(__texMemAlpha, __pAlpha);
		}

		/// <summary>
		/// GPU 메모리를 반환하고, 텍스처 인터페이스와 바인딩을 해제한다.
		/// </summary>
		static void __free()
		{
			if (__pRed)
			{
				cudaUnbindTexture(__texMemRed);
				cudaFreeArray(__pRed);

				__pRed = nullptr;
			}

			if (__pGreen)
			{
				cudaUnbindTexture(__texMemGreen);
				cudaFreeArray(__pGreen);

				__pGreen = nullptr;
			}

			if (__pBlue)
			{
				cudaUnbindTexture(__texMemBlue);
				cudaFreeArray(__pBlue);

				__pBlue = nullptr;
			}

			if (__pAlpha)
			{
				cudaUnbindTexture(__texMemAlpha);
				cudaFreeArray(__pAlpha);

				__pAlpha = nullptr;
			}
		}
	}



	/* function */
	/// <summary>
	/// 스크린의 수평, 수직 방향으로 offset을 이동하여 픽셀의 위치를 알아낸다.
	/// </summary>
	/// <param name="screenWidth">
	/// 스크린 너비
	/// </param>
	/// <param name="screenHeight">
	/// 스크린 높이
	/// </param>
	/// <param name="W_IDX">
	/// 수평 방향 인덱스
	/// </param>
	/// <param name="H_IDX">
	/// 수직 방향 인덱스
	/// </param>
	/// <param name="camPosition">
	/// 카메라 위치
	/// </param>
	/// <param name="orthoBasis">
	/// 직교 기저
	/// </param>
	/// <param name="imgBasedSamplingStep">
	/// 스크린 offset 이동 단위
	/// </param>
	/// <returns>
	/// 픽셀 위치
	/// </returns>
	__device__
	static Point3D __calcStartingPoint(
		const int screenWidth, const int screenHeight,
		const int wIdx, const int hIdx,
		const Point3D& camPosition, const OrthoBasis& orthoBasis,
		const float imgBasedSamplingStep)
	{
		const int SCR_WIDTH_HALF = (screenWidth / 2);
		const int SCR_HEIGHT_HALF = (screenHeight / 2);

		const float SCR_HORIZ_OFFSET = (static_cast<float>(wIdx - SCR_WIDTH_HALF) * imgBasedSamplingStep);
		const float SCR_VERT_OFFSET = (static_cast<float>(SCR_HEIGHT_HALF - hIdx) * imgBasedSamplingStep);

		Point3D samplePoint = camPosition;
		samplePoint += (orthoBasis.v * SCR_HORIZ_OFFSET);
		samplePoint += (orthoBasis.w * SCR_VERT_OFFSET);

		return samplePoint;
	}

	/// <summary>
	/// 주어진 샘플 위치에서 법선 벡터를 계산하여 반환한다.
	/// </summary>
	/// <param name="samplePoint">
	/// 샘플 위치
	/// </param>
	/// <returns>
	/// 법선 벡터
	/// </returns>
	__device__ 
	static Vector3D __getNormal(const Point3D& samplePoint)
	{
		const Point3D ADJ_POINT =
		{
			samplePoint.x + .5f,
			samplePoint.y + .5f,
			samplePoint.z + .5f
		};

		const float X_LEFT = tex3D(Volume::__texMem, (ADJ_POINT.x - 1.f), ADJ_POINT.y, ADJ_POINT.z);
		const float X_RIGHT = tex3D(Volume::__texMem, (ADJ_POINT.x + 1.f), ADJ_POINT.y, ADJ_POINT.z);

		const float Y_LEFT = tex3D(Volume::__texMem, ADJ_POINT.x, (ADJ_POINT.y - 1.f), ADJ_POINT.z);
		const float Y_RIGHT = tex3D(Volume::__texMem, ADJ_POINT.x, (ADJ_POINT.y + 1.f), ADJ_POINT.z);

		const float Z_LEFT = tex3D(Volume::__texMem, ADJ_POINT.x, ADJ_POINT.y, (ADJ_POINT.z - 1.f));
		const float Z_RIGHT = tex3D(Volume::__texMem, ADJ_POINT.x, ADJ_POINT.y, (ADJ_POINT.z + 1.f));

		return NumberUtility::inverseGradient(
			X_LEFT, X_RIGHT, Y_LEFT, Y_RIGHT, Z_LEFT, Z_RIGHT);
	}

	/// <summary>
	/// 샘플의 색상 정보를 결정한다.
	/// </summary>
	/// <param name="samplePoint">
	/// 샘플 위치
	/// </param>
	/// <param name="INTENSITY">
	/// 샘플 위치에서의 볼륨 데이터 값
	/// </param>
	/// <param name="alpha">
	/// 샘플 위치에서의 alpha 값
	/// </param>
	/// <param name="shininess">
	/// 광택 정도
	/// </param>
	/// <param name="light1">
	/// 1번 조명 레퍼런스
	/// </param>
	/// <param name="light2">
	/// 2번 조명 레퍼런스
	/// </param>
	/// <param name="light3">
	/// 3번 조명 레퍼런스
	/// </param>
	/// <param name="camDirection">
	/// 카메라 시점 벡터
	/// </param>
	/// <returns></returns>
	__device__ static Color<float> __shade(
		const Point3D& samplePoint,
		const float intensity, const float alpha, const float shininess,
		const Light& light1, const Light& light2, const Light& light3, 
		const Vector3D& camDirection)
	{
		// 볼륨 데이터 값을 transfer function에 넣어 albedo 정보를 가져온다.
		const float RED = tex1D(TransferFunc::__texMemRed, intensity + .5f);
		const float GREEN = tex1D(TransferFunc::__texMemGreen, intensity + .5f);
		const float BLUE = tex1D(TransferFunc::__texMemBlue, intensity + .5f);

		const Color<float> ALBEDO(RED, GREEN, BLUE);

		// 조명이 활성화되어 있지 않다면 albedo를 리턴한다.
		if (!light1.enabled && !light1.enabled && !light2.enabled)
			return ALBEDO;

		// 조명을 반영한다.
		const Light LIGHTS[3] = { light1, light1, light2 };
		Color<float> retVal(0.f);

		for (int i = 0; i < 3; ++i)
		{
			if (LIGHTS[i].enabled)
			{
				const Vector3D N = __getNormal(samplePoint);
				const Vector3D L = (LIGHTS[i].position - samplePoint).getUnit();
				const Vector3D V = -camDirection;
				const Vector3D H = ((L + V) * .5f);

				// ambient
				retVal += (ALBEDO * LIGHTS[i].ambient);

				// diffuse
				float dotNL = N.dot(L);
				if (dotNL > 0.f)
					retVal += (ALBEDO * LIGHTS[i].diffuse * dotNL);

				// specular
				float dotNH = N.dot(H);
				if (dotNH > 0.f)
					retVal += (LIGHTS[i].specular * powf(dotNH, (shininess * 2.f)));
			}
		}

		return retVal;
	}

	/// <summary>
	/// 시야 광선을 투사한 뒤 누적 결과를 스크린에 기록한다.
	/// </summary>
	/// <param name="screenWidth">
	/// 스크린 너비
	/// </param>
	/// <param name="screenHeight">
	/// 스크린 높이
	/// </param>
	/// <param name="camPosition">
	/// 카메라 위치
	/// </param>
	/// <param name="orthoBasis">
	/// 직교 기저
	/// </param>
	/// <param name="light3">
	/// 1번 조명
	/// </param>
	/// <param name="light3">
	/// 2번 조명
	/// </param>
	/// <param name="light3">
	/// 3번 조명
	/// </param>
	/// <param name="shininess">
	/// 광택 정도
	/// </param>
	/// <param name="imgBasedSamplingStep">
	/// 스크린 offset 이동 단위
	/// </param>
	/// <param name="objectBasedSamplingStep">
	/// 시야 광선 전진 단위
	/// </param>
	/// <param name="pScreen">
	/// 스크린 포인터
	/// </param>
	__global__
	static void __raycast(
		const int screenWidth, const int screenHeight,
		const Point3D camPosition, const OrthoBasis orthoBasis,
		const Light light0, const Light light1, const Light light2, const float shininess,
		const float imgBasedSamplingStep, const float objectBasedSamplingStep, Pixel* pScreen)
	{
		// 스크린은 논리적으로 2차원이나, 물리적으로는 1차원이다.
		// 따라서 스크린 픽셀에 접근하기 위해서는
		// 수평, 수직 방향 인덱스를 계산하여 offset을 구해야 한다.
		const int W_IDX = ((blockIdx.x * blockDim.x) + threadIdx.x);
		const int H_IDX = ((blockIdx.y * blockDim.y) + threadIdx.y);
		const int OFFSET = ((screenWidth * H_IDX) + W_IDX);

		// 샘플의 시작점을 픽셀 위치로 설정한다.
		Point3D samplePoint = __calcStartingPoint(
			screenWidth, screenHeight, W_IDX, H_IDX, camPosition, orthoBasis, imgBasedSamplingStep);

		// 픽셀 위치에서 카메라 시점 방향으로 시야 광선을 투사하였을 때 
		// 볼륨을 투과하는지 여부를 조사하고, 투과 영역을 알아낸다.
		const Range<float> RANGE = RayBoxIntersector::getValidRange(
			Volume::__meta->size, samplePoint, orthoBasis.u);

		// 시야 광선이 볼륨을 투과하지 않는다면, 픽셀 값을 0으로 설정하고 리턴한다.
		if (RANGE.end < RANGE.start)
		{
			pScreen[OFFSET].set(0.f);
			return;
		}

		// 샘플 위치를 투과 영역 시작 지점까지 이동한다.
		samplePoint += (orthoBasis.u * RANGE.start);

		float transparency = 1.f; // 누적 투명도
		Color<float> color(0.f); // 누적 RGB 세기

		const Vector3D STEP_VECTOR = (orthoBasis.u * objectBasedSamplingStep);
		for (float t = RANGE.start; t < RANGE.end; t += objectBasedSamplingStep)
		{
			// 현재 샘플 위치에서 볼륨 데이터 값을 선형 보간하여 가져온다.
			// 0.5를 더하는 것은 CUDA의 텍스처 메모리 특성 때문. 지우지 말 것
			const float INTENSITY = tex3D(
				Volume::__texMem, samplePoint.x + .5f, samplePoint.y + .5f, samplePoint.z + .5f);

			// 볼륨 데이터 값을 보정한다.
			const float CORRECTED_INTENSITY = (INTENSITY * USHRT_MAX);

			// alpha transfer function에 위의 값을 넣어 alpha 값을 알아낸다.
			// 세인 변경: 이제 normalized 인덱스가 아님. 실제 값을 넣자 (RGB transfer function 모두에게 해당됨)
			const float ALPHA = tex1D(TransferFunc::__texMemAlpha, CORRECTED_INTENSITY + .5f);

			// 시야 광선 전진 단위에 따라 alpha 값을 보정한다.
			// (전진 단위가 작아지면 alpha 값이 금방 누적되어 원하는 결과를 얻을 수 없다.)
			const float CORRECTED_ALPHA = (1.f - powf((1.f - ALPHA), objectBasedSamplingStep));

			// alpha가 0이 아닌 경우에 한해서
			if (!NumberUtility::nearEqual(ALPHA, 0.f))
			{
				// shading 연산을 수행한다.
				const Color<float> SHADING_RESULT = __shade(
					samplePoint, CORRECTED_INTENSITY, CORRECTED_ALPHA, shininess, light0, light1, light2, orthoBasis.u);

				// alpha-blending
				const float CUR_ALPHA = (transparency * CORRECTED_ALPHA);

				color.red += (CUR_ALPHA * SHADING_RESULT.red);
				color.green += (CUR_ALPHA * SHADING_RESULT.green);
				color.blue += (CUR_ALPHA * SHADING_RESULT.blue);

				transparency *= (1.f - CORRECTED_ALPHA);

				// 누적 투명도가 0.01f보다 작아지면 
				// 시야 광선의 전진을 종료한다. (early-ray termination)
				if (transparency < 0.01f)
					break;
			}

			// 시야 광선을 전진한다.
			samplePoint += STEP_VECTOR;
		}

		// 픽셀 값을 alpha-blending compositing 연산 결과로 설정한다.
		pScreen[OFFSET].set(
			static_cast<ubyte>(NumberUtility::truncate(color.red * 255.f, 0.f, 255.f)),
			static_cast<ubyte>(NumberUtility::truncate(color.green * 255.f, 0.f, 255.f)),
			static_cast<ubyte>(NumberUtility::truncate(color.blue * 255.f, 0.f, 255.f)));
	}
}

////////////////////////////////
//// RENDERING ENGINE START ////
////////////////////////////////

/* constructor */
RenderingEngine::RenderingEngine() :
	volumeRenderer(__volumeMeta, __initialized), imageProcessor(__volumeMeta, __initialized)
{}

/* member function */
void RenderingEngine::loadVolume(const VolumeData& volumeData)
{
	__volumeMeta = volumeData.meta;

	Device::Volume::__free();
	Device::Volume::__malloc(volumeData);

	volumeRenderer.__onLoadVolume();
	imageProcessor.__onLoadVolume();

	__initialized = true;
}

void RenderingEngine::onSystemInit()
{
	SystemIndirectAccessor::getEventBroadcaster().addVolumeLoadingListener(*this);
}

void RenderingEngine::onLoadVolume(const VolumeData& volumeData)
{
	loadVolume(volumeData);
}

RenderingEngine& RenderingEngine::getInstance()
{
	return __instance;
}

///////////////////////////////
//// VOLUME RENDERER START ////
///////////////////////////////

/* constructor */
RenderingEngine::VolumeRenderer::VolumeRenderer(const VolumeMeta &volumeMeta, const bool &initFlag) :
	__volumeMeta(volumeMeta), __initialized(initFlag)
{}

/* member function */
void RenderingEngine::VolumeRenderer::render(
	Pixel *const pScreen, const int screenWidth, const int screenHeight)
{
	if (__initialized)
	{
		const OrthoBasis ORTHOBASIS = camera.getOrthoBasis();

		Device::__raycast<<<dim3(screenWidth / 16, screenHeight / 16), dim3(16, 16)>>>(
			screenWidth, screenHeight,
			camera.getPosition(), ORTHOBASIS,
			__light[0], __light[1], __light[2], __shininess,
			__imgBasedSamplingStep, __objectBasedSamplingStep, pScreen);
	}
}

void RenderingEngine::VolumeRenderer::adjustImgBasedSamplingStep(const float delta) 
{
	__imgBasedSamplingStep += delta;

	if (__imgBasedSamplingStep < 0.1f)
		__imgBasedSamplingStep = 0.1f;
	else if (__imgBasedSamplingStep > 4.f)
		__imgBasedSamplingStep = 4.f;
}

void RenderingEngine::VolumeRenderer::__onLoadVolume()
{
	// 새로운 볼륨이 렌더링 엔진에 적재된 후 나중에 호출되는 콜백 함수
	// 새로운 볼륨 적재 시 필요한 처리 루틴 작성
	Device::TransferFunc::__free();
	__initTransferFunc();
	__initLight();
	__initCamera();
}

void RenderingEngine::VolumeRenderer::__initTransferFunc()
{
	__pTransferFunc = new TransferFunction(__volumeMeta.voxelPrecision);

	/*__pTransferFunc->setRed({ 0, __volumeMeta.voxelPrecision });
	__pTransferFunc->setGreen({ 0, __volumeMeta.voxelPrecision });
	__pTransferFunc->setBlue({ 0, __volumeMeta.voxelPrecision });
	__pTransferFunc->setAlpha({ 0, __volumeMeta.voxelPrecision });*/

	ushort minVal = 3300;
	ushort maxVal = 4000;

	__pTransferFunc->setRed({ minVal, maxVal });
	__pTransferFunc->setGreen({ minVal, maxVal });
	__pTransferFunc->setBlue({ minVal, maxVal });
	__pTransferFunc->setAlpha({ minVal, maxVal });

	Device::TransferFunc::__malloc(*__pTransferFunc);
}

void RenderingEngine::VolumeRenderer::__initLight()
{
	__light[0].position = Constant::Light::Position::LEFT;
	__light[1].position = Constant::Light::Position::RIGHT;
	__light[2].position = Constant::Light::Position::BACK;

	__light[0].ambient = { 0.03f, 0.02f, 0.01f };
	__light[0].diffuse = { 0.4f, 0.1f, 0.1f };
	__light[0].specular = { 0.5f, 0.1f, 0.1f };

	__light[1].ambient = { 0.02f, 0.03f, 0.01f };
	__light[1].diffuse = { 0.1f, 0.6f, 0.1f };
	__light[1].specular = { 0.1f, 0.5f, 0.1f };

	__light[2].ambient = { 0.01f, 0.02f, 0.03f };
	__light[2].diffuse = { 0.2f, 0.4f, 0.6f };
	__light[2].specular = { 0.3f, 0.4f, 0.5f };
}

void RenderingEngine::VolumeRenderer::__initCamera()
{
	camera.set(Constant::Camera::EYE, Constant::Camera::AT, Constant::Camera::UP);
}

///////////////////////////////
//// IMAGE PROCESSOR START ////
///////////////////////////////

namespace __Device
{
	namespace __ImgProc
	{
		texture<float> __transferTable;

		__global__
		void __kernel_imgProcRender_top(
			Pixel* const pScreen, const Size2D<> screenSize,
			const float samplingStep, const float volumeSlicingZ)
		{
			using Device::Volume::__meta;

			const Index2D<> SCR_IDX = Index2D<>::getKernelIndex();

			const int TEX_Y = ((screenSize.height - 1) - SCR_IDX.y);
			const int SCR_OFFSET = ((TEX_Y * screenSize.width) + SCR_IDX.x);

			const float VOL_X =
				(static_cast<float>(SCR_IDX.x) +
				(static_cast<float>(__meta->size.width - screenSize.width) * .5f));

			const float VOL_Y =
				(static_cast<float>((__meta->size.height - 1) - SCR_IDX.y) -
				(static_cast<float>(__meta->size.height - screenSize.height) * .5f));

			const float INTENSITY =
				(tex3D(Device::Volume::__texMem, VOL_X + .5f, VOL_Y + .5f, volumeSlicingZ + .5f) *
				static_cast<float>(USHRT_MAX));

			const float RET_VAL = tex1D(__transferTable, INTENSITY + .5f);

			pScreen[SCR_OFFSET].set(
				static_cast<ubyte>(NumberUtility::truncate(RET_VAL * 255.f, 0.f, 255.f)));
		}

		__global__
		void __kernel_imgProcRender_front(
			Pixel* const pScreen, const Size2D<> screenSize,
			const float samplingStep, const float volumeSlicingY)
		{
			using Device::Volume::__meta;

			Index2D<> SCR_IDX = Index2D<>::getKernelIndex();
			SCR_IDX.y = ((screenSize.height - 1) - SCR_IDX.y);

			const int TEX_Y = ((screenSize.height - 1) - SCR_IDX.y);
			const int SCR_OFFSET = ((TEX_Y * screenSize.width) + SCR_IDX.x);

			const float VOL_X =
				(static_cast<float>(SCR_IDX.x) +
				(static_cast<float>(__meta->size.width - screenSize.width) * .5f));

			const float VOL_Z =
				(static_cast<float>(SCR_IDX.y) +
				(static_cast<float>(__meta->size.depth - screenSize.height) * .5f));

			const float INTENSITY =
				(tex3D(Device::Volume::__texMem, VOL_X + .5f, volumeSlicingY + .5f, VOL_Z + .5f) *
				static_cast<float>(USHRT_MAX));

			const float RET_VAL = tex1D(__transferTable, INTENSITY + .5f);

			pScreen[SCR_OFFSET].set(
				static_cast<ubyte>(NumberUtility::truncate(RET_VAL * 255.f, 0.f, 255.f)));
		}

		__global__
		void __kernel_imgProcRender_right(
			Pixel* const pScreen, const Size2D<> screenSize,
			const float samplingStep, const float volumeSlicingX)
		{
			using Device::Volume::__meta;

			const Index2D<> SCR_IDX = Index2D<>::getKernelIndex();

			const int TEX_Y = ((screenSize.height - 1) - SCR_IDX.y);
			const int SCR_OFFSET = ((TEX_Y * screenSize.width) + SCR_IDX.x);

			const float VOL_Y =
				(static_cast<float>(SCR_IDX.x) +
				(static_cast<float>(__meta->size.height - screenSize.width) * .5f));

			const float VOL_Z =
				(static_cast<float>(SCR_IDX.y) +
				(static_cast<float>(__meta->size.depth - screenSize.height) * .5f));

			const float INTENSITY =
				(tex3D(Device::Volume::__texMem, volumeSlicingX + .5f, VOL_Y + .5f, VOL_Z + .5f) *
				static_cast<float>(USHRT_MAX));

			const float RET_VAL = tex1D(__transferTable, INTENSITY + .5f);

			pScreen[SCR_OFFSET].set(
				static_cast<ubyte>(NumberUtility::truncate(RET_VAL * 255.f, 0.f, 255.f)));
		}
	}
}

RenderingEngine::ImageProcessor::ImageProcessor(const VolumeMeta &volumeMeta, const bool &initFlag) :
	__volumeMeta(volumeMeta), __initialized(initFlag)
{
	using namespace __Device::__ImgProc;
	
	__transferTable.filterMode = cudaTextureFilterMode::cudaFilterModeLinear;
}

void RenderingEngine::ImageProcessor::__onLoadVolume()
{
	__init();
}

void RenderingEngine::ImageProcessor::__sync()
{
	if (__transferTableDirty)
		__syncTransferFunction();
}

void RenderingEngine::ImageProcessor::__syncTransferFunction()
{
	const float RANGE_INV =
		(1.f / static_cast<float>(_transferTableBoundary.end - _transferTableBoundary.start));

	const Range<int> BOUNDARY = _transferTableBoundary.castTo<int>();

	const int ITER = __volumeMeta.voxelPrecision;
	for (int i = 0; i < ITER; i++)
	{
		if (i < BOUNDARY.start)
			__pHostTransferTableBuffer[i] = 0.f;
		else if (i < BOUNDARY.end)
			__pHostTransferTableBuffer[i] = (static_cast<float>(i - BOUNDARY.start) * RANGE_INV);
		else
			__pHostTransferTableBuffer[i] = 1.f;
	}

	cudaMemcpyToArray(
		__transferTableBuffer, 0, 0, __pHostTransferTableBuffer,
		__volumeMeta.voxelPrecision * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);

	__transferTableDirty = false;
}

void RenderingEngine::ImageProcessor::__init()
{
	__release();

	__pHostTransferTableBuffer = new float[__volumeMeta.voxelPrecision];

	const cudaChannelFormatDesc DESC = cudaCreateChannelDesc<float>();

	cudaMallocArray(
		&__transferTableBuffer, &DESC, __volumeMeta.voxelPrecision * sizeof(float));

	using namespace __Device::__ImgProc;
	cudaBindTextureToArray(__transferTable, __transferTableBuffer);

	const Size3D<float> DIVISION_POINT = (__volumeMeta.size.castTo<float>() * .5f);
	__slicingPoint.set(
		DIVISION_POINT.width, DIVISION_POINT.height, DIVISION_POINT.depth);

	_transferTableBoundary.set(0, __volumeMeta.voxelPrecision);
	__transferTableDirty = true;
}

void RenderingEngine::ImageProcessor::__release()
{

	if (__transferTableBuffer)
	{
		using namespace __Device::__ImgProc;
		cudaUnbindTexture(__transferTable);

		cudaFreeArray(__transferTableBuffer);
		__transferTableBuffer = nullptr;

		delete[] __pHostTransferTableBuffer;
		__pHostTransferTableBuffer = nullptr;
	}
}

void RenderingEngine::ImageProcessor::setTransferFunction(const ushort startInc, const ushort endExc)
{
	_transferTableBoundary.set(startInc, endExc);
	__transferTableDirty = true;
}

void RenderingEngine::ImageProcessor::setTransferFunction(const Range<ushort> &range)
{
	setTransferFunction(range.start, range.end);
}

void RenderingEngine::ImageProcessor::render(
	Pixel *const pScreen, const int screenWidth, const int screenHeight,
	const float samplingStep, const SliceAxis axis)
{
	if (__initialized)
	{
		__sync();

		using namespace __Device::__ImgProc;

		const dim3 gridDim =
		{
			static_cast<uint>(screenWidth / 16),
			static_cast<uint>(screenHeight / 16)
		};
		const dim3 blockDim = { 16, 16 };

		switch (axis)
		{
		case SliceAxis::TOP:
			__kernel_imgProcRender_top<<<gridDim, blockDim>>>(
				pScreen, { screenWidth, screenHeight }, samplingStep, __slicingPoint.z);
			break;

		case SliceAxis::FRONT:
			__kernel_imgProcRender_front<<<gridDim, blockDim>>>(
				pScreen, { screenWidth, screenHeight }, samplingStep, __slicingPoint.y);
			break;

		case SliceAxis::RIGHT:
			__kernel_imgProcRender_right<<<gridDim, blockDim>>>(
				pScreen, { screenWidth, screenHeight }, samplingStep, __slicingPoint.x);
			break;
		}
	}
}

RenderingEngine::ImageProcessor::~ImageProcessor()
{
	__release();
}