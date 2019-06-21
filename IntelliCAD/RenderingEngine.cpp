/*
*	Copyright (C) 2019 Jin Won. All right reserved.
*
*	���ϸ�			: RenderingEngine.cpp
*	�ۼ���			: ����
*	���� ������		: 19.04.07
*/

#include "custom_cudart.h"
#include "RenderingEngine.hpp"
#include "RayBoxInstersector.h"
#include "NumberUtility.hpp"
#include "Constant.h"
#include <limits>
#include "Index2D.hpp"
#include "VolumeIndicator.h"

RenderingEngine RenderingEngine::__instance;

using namespace std;

namespace __Device
{
	static Light *__pLightsDev;

	// __Volume Indicator
	__device__
	static float __indicatorLength[] = { 70.f };

	__device__
	static float __indicatorThickness[] = { 6.f };

	__device__
	static float __indicatorAlpha[] = { .3f };

	__device__
	static Color<float> __indicatorColor[] =
	{{ (220.f / 256.f), (25.f / 256.f), (72.f / 256.f) }};

	namespace __Volume
	{
		/* variable */
		/// <summary>
		/// <para>GPU �޸𸮸� �Ҵ��ϴ� ���� ��Ÿ ������</para>
		/// <para>���� ũ��, ������ ����, ���� ���е��� ������.</para>
		/// </summary>
		__device__
		static VolumeNumericMeta __meta[1];

		/// <summary>
		/// <para>GPU �޸𸮸� �Ҵ��ϴ� short Ÿ�� ���� 3���� �迭</para>
		/// <para>�����͸� 0.f~1.f ���̷� ����ȭ�ؼ� �����Ѵ�.</para>
		/// </summary>
		static cudaArray_t __pBuffer = nullptr;

		/// <summary>
		/// <para>GPU ���� 3���� �迭�� ���ε� �� �ؽ�ó �������̽�</para>
		/// <para>float Ÿ�� 3���� index�� �����Ͽ� �����Ǵ� ���� �����͸� ���� �����ؼ� �����´�.</para>
		/// <para>���� ����: tex3D(texMem, x, y, z);</para>
		/// </summary>
		static texture<ushort, 3, cudaReadModeNormalizedFloat> __volTexture;

		/* function */
		/// <summary>
		/// GPU �޸𸮸� �Ҵ��ϰ�, �ؽ�ó �������̽��� ���ε� �Ѵ�.
		/// </summary>
		/// <param name="volumeData">
		/// <para>CPU ���� ������ ���۷���</para>
		/// <para>��Ÿ �����͸� �����Ѵ�.</para>
		/// </param>
		static void __malloc(const VolumeData& volumeData)
		{
			const cudaChannelFormatDesc DESC = cudaCreateChannelDesc<ushort>();
			const VolumeNumericMeta &numericMeta = volumeData.meta.numeric;

			// ���� ��Ÿ �����͸� CPU���� GPU ������ �����Ѵ�.  
			cudaMemcpyToSymbol(__meta, &numericMeta, sizeof(VolumeNumericMeta));

			// ������ ũ��
			const cudaExtent MEM_EXTENT =
				make_cudaExtent(numericMeta.memSize.width, numericMeta.memSize.height, numericMeta.memSize.depth);

			// ���� 3���� �迭 �޸𸮸� GPU�� �Ҵ��Ѵ�.
			cudaMalloc3DArray(&__pBuffer, &DESC, MEM_EXTENT);

			// �޸� ���� �Ķ���͸� �����Ѵ�.
			cudaMemcpy3DParms params;
			ZeroMemory(&params, sizeof(cudaMemcpy3DParms));

			params.srcPtr = make_cudaPitchedPtr(
				volumeData.pBuffer.get(), (MEM_EXTENT.width * sizeof(ushort)), MEM_EXTENT.width, MEM_EXTENT.height);
			
			params.dstArray = __pBuffer;
			params.extent = MEM_EXTENT;

			// ���� �����͸� CPU���� GPU �迭�� �����Ѵ�.
			cudaMemcpy3D(&params);

			__volTexture.addressMode[0] = cudaAddressModeBorder;
			__volTexture.addressMode[1] = cudaAddressModeBorder;
			__volTexture.addressMode[2] = cudaAddressModeBorder;
			__volTexture.filterMode = cudaFilterModeLinear;

			// �ؽ�ó �����ڿ� ���� 3���� �迭�� ���ε��Ѵ�.
			cudaBindTextureToArray(__volTexture, __pBuffer);
		}

		/// <summary>
		/// GPU �޸𸮸� ��ȯ�ϰ�, �ؽ�ó �������̽��� ���ε��� �����Ѵ�.
		/// </summary>
		static void __free()
		{
			if (__pBuffer)
			{
				cudaUnbindTexture(__volTexture);
				cudaFreeArray(__pBuffer);

				__pBuffer = nullptr;
			}
		}
	}

	/// <summary>
	/// <para>GPU red ���Ϳ� ���ε� �� �ؽ�ó ������</para>
	/// <para>0.f~1.f ���̷� ����ȭ �� float Ÿ�� 1���� index�� ������ �� �ִ�.</para>
	/// <para>�̶� �����Ǵ� ���� red ���͸� ���� �����Ͽ� �����´�.</para>
	/// </summary>
	static texture<float> __transferTableRed;

	/// <summary>
	/// <para>GPU green ���Ϳ� ���ε� �� �ؽ�ó ������</para>
	/// <para>0.f~1.f ���̷� ����ȭ �� float Ÿ�� 1���� index�� ������ �� �ִ�.</para>
	/// <para>�̶� �����Ǵ� ���� green ���͸� ���� �����Ͽ� �����´�.</para>
	/// </summary>
	static texture<float> __transferTableGreen;

	/// <summary>
	/// <para>GPU blue ���Ϳ� ���ε� �� �ؽ�ó ������</para>
	/// <para>0.f~1.f ���̷� ����ȭ �� float Ÿ�� 1���� index�� ������ �� �ִ�.</para>
	/// <para>�̶� �����Ǵ� ���� blue ���͸� ���� �����Ͽ� �����´�.</para>
	/// </summary>
	static texture<float> __transferTableBlue;

	/// <summary>
	/// <para>GPU alpha ���Ϳ� ���ε� �� �ؽ�ó ������</para>
	/// <para>0.f~1.f ���̷� ����ȭ �� float Ÿ�� 1���� index�� ������ �� �ִ�.</para>
	/// <para>�̶� �����Ǵ� ���� alpha ���͸� ���� �����Ͽ� �����´�.</para>
	/// </summary>
	static texture<float> __transferTableAlpha;

	/* function */
	/// <summary>
	/// ��ũ���� ����, ���� �������� offset�� �̵��Ͽ� �ȼ��� ��ġ�� �˾Ƴ���.
	/// </summary>
	/// <param name="screenWidth">
	/// ��ũ�� �ʺ�
	/// </param>
	/// <param name="screenHeight">
	/// ��ũ�� ����
	/// </param>
	/// <param name="W_IDX">
	/// ���� ���� �ε���
	/// </param>
	/// <param name="H_IDX">
	/// ���� ���� �ε���
	/// </param>
	/// <param name="camPosition">
	/// ī�޶� ��ġ
	/// </param>
	/// <param name="orthoBasis">
	/// ���� ����
	/// </param>
	/// <param name="imgBasedSamplingStep">
	/// ��ũ�� offset �̵� ����
	/// </param>
	/// <returns>
	/// �ȼ� ��ġ
	/// </returns>
	__device__
	static Point3D __calcStartingPoint(
		const int screenWidth, const int screenHeight,
		const int wIdx, const int hIdx,
		const Point3D& camPosition, const OrthoBasis& orthoBasis,
		const float imgBasedSamplingStep)
	{
		const float SCR_WIDTH_HALF = (static_cast<float>(screenWidth) * .5f);
		const float SCR_HEIGHT_HALF = (static_cast<float>(screenHeight) * .5f);

		const float SCR_HORIZ_OFFSET = ((static_cast<float>(wIdx) - SCR_WIDTH_HALF) * imgBasedSamplingStep);
		const float SCR_VERT_OFFSET = ((SCR_HEIGHT_HALF - static_cast<float>(hIdx)) * imgBasedSamplingStep);

		Point3D samplePoint = camPosition;
		samplePoint += (orthoBasis.v * SCR_HORIZ_OFFSET);
		samplePoint += (orthoBasis.w * SCR_VERT_OFFSET);

		return samplePoint;
	}

	/// <summary>
	/// �־��� ���� ��ġ���� ���� ���͸� ����Ͽ� ��ȯ�Ѵ�.
	/// </summary>
	/// <param name="samplePoint">
	/// ���� ��ġ
	/// </param>
	/// <returns>
	/// ���� ����
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

		const float X_LEFT = tex3D(__Volume::__volTexture, (ADJ_POINT.x - 1.f), ADJ_POINT.y, ADJ_POINT.z);
		const float X_RIGHT = tex3D(__Volume::__volTexture, (ADJ_POINT.x + 1.f), ADJ_POINT.y, ADJ_POINT.z);

		const float Y_LEFT = tex3D(__Volume::__volTexture, ADJ_POINT.x, (ADJ_POINT.y - 1.f), ADJ_POINT.z);
		const float Y_RIGHT = tex3D(__Volume::__volTexture, ADJ_POINT.x, (ADJ_POINT.y + 1.f), ADJ_POINT.z);

		const float Z_LEFT = tex3D(__Volume::__volTexture, ADJ_POINT.x, ADJ_POINT.y, (ADJ_POINT.z - 1.f));
		const float Z_RIGHT = tex3D(__Volume::__volTexture, ADJ_POINT.x, ADJ_POINT.y, (ADJ_POINT.z + 1.f));

		return NumberUtility::inverseGradient(
			X_LEFT, X_RIGHT, Y_LEFT, Y_RIGHT, Z_LEFT, Z_RIGHT);
	}

	/// <summary>
	/// ������ ���� ������ �����Ѵ�.
	/// </summary>
	/// <param name="samplePoint">
	/// ���� ��ġ
	/// </param>
	/// <param name="INTENSITY">
	/// ���� ��ġ������ ���� ������ ��
	/// </param>
	/// <param name="alpha">
	/// ���� ��ġ������ alpha ��
	/// </param>
	/// <param name="shininess">
	/// ���� ����
	/// </param>
	/// <param name="light1">
	/// 1�� ���� ���۷���
	/// </param>
	/// <param name="light2">
	/// 2�� ���� ���۷���
	/// </param>
	/// <param name="light3">
	/// 3�� ���� ���۷���
	/// </param>
	/// <param name="camDirection">
	/// ī�޶� ���� ����
	/// </param>
	/// <returns></returns>
	__device__ static Color<float> __shade(
		const Point3D& samplingPoint_mappedToMem,
		const float intensity, const float alpha, const float shininess,
		const Light *const pLights, const Vector3D& camDirection)
	{
		// ���� ������ ���� transfer function�� �־� albedo ������ �����´�.
		const float RED = tex1D(__transferTableRed, intensity + .5f);
		const float GREEN = tex1D(__transferTableGreen, intensity + .5f);
		const float BLUE = tex1D(__transferTableBlue, intensity + .5f);

		const Color<float> ALBEDO(RED, GREEN, BLUE);

		// ������ Ȱ��ȭ�Ǿ� ���� �ʴٸ� albedo�� �����Ѵ�.
		if (!pLights[0].enabled && !pLights[1].enabled && !pLights[2].enabled)
			return ALBEDO;

		// ������ �ݿ��Ѵ�.
		Color<float> retVal(0.f);

		for (int i = 0; i < 3; ++i)
		{
			if (pLights[i].enabled)
			{
				const Vector3D N = __getNormal(samplingPoint_mappedToMem);
				const Vector3D L = (pLights[i].position - samplingPoint_mappedToMem).getUnit();
				const Vector3D V = -camDirection;
				const Vector3D H = ((L + V) * .5f);

				// ambient
				retVal += (ALBEDO * pLights[i].ambient);

				// diffuse
				float dotNL = N.dot(L);
				if (dotNL > 0.f)
					retVal += (ALBEDO * pLights[i].diffuse * dotNL);

				// specular
				float dotNH = N.dot(H);
				if (dotNH > 0.f)
					retVal += (pLights[i].specular * powf(dotNH, (shininess * 2.f)));
			}
		}

		return retVal;
	}

	__device__
	static Point3D __mapIdxVolumeToMem(const Point3D &volumeIdx)
	{
		return
		{
			volumeIdx.x / __Volume::__meta->spacing.width,
			volumeIdx.y / __Volume::__meta->spacing.height,
			volumeIdx.z / __Volume::__meta->spacing.depth,
		};
	}

	/// <summary>
	/// �þ� ������ ������ �� ���� ����� ��ũ���� ����Ѵ�.
	/// </summary>
	/// <param name="screenWidth">
	/// ��ũ�� �ʺ�
	/// </param>
	/// <param name="screenHeight">
	/// ��ũ�� ����
	/// </param>
	/// <param name="camPosition">
	/// ī�޶� ��ġ
	/// </param>
	/// <param name="orthoBasis">
	/// ���� ����
	/// </param>
	/// <param name="light3">
	/// 1�� ����
	/// </param>
	/// <param name="light3">
	/// 2�� ����
	/// </param>
	/// <param name="light3">
	/// 3�� ����
	/// </param>
	/// <param name="shininess">
	/// ���� ����
	/// </param>
	/// <param name="imgBasedSamplingStep">
	/// ��ũ�� offset �̵� ����
	/// </param>
	/// <param name="objectBasedSamplingStep">
	/// �þ� ���� ���� ����
	/// </param>
	/// <param name="pScreen">
	/// ��ũ�� ������
	/// </param>
	__global__
	static void __raycast(
		const int screenWidth, const int screenHeight,
		const Point3D camPosition, const OrthoBasis orthoBasis,
		const Light *const pLightsDev, const float shininess,
		const float imgBasedSamplingStep, const float objectBasedSamplingStep, Pixel* pScreen)
	{
		// ��ũ���� �������� 2�����̳�, ���������δ� 1�����̴�.
		// ���� ��ũ�� �ȼ��� �����ϱ� ���ؼ���
		// ����, ���� ���� �ε����� ����Ͽ� offset�� ���ؾ� �Ѵ�.
		const int W_IDX = ((blockIdx.x * blockDim.x) + threadIdx.x);
		const int H_IDX = ((blockIdx.y * blockDim.y) + threadIdx.y);
		const int OFFSET = ((screenWidth * H_IDX) + W_IDX);

		// ������ �������� �ȼ� ��ġ�� �����Ѵ�.
		const Point3D STARTING_POINT = __calcStartingPoint(
			screenWidth, screenHeight, W_IDX, H_IDX, camPosition, orthoBasis, imgBasedSamplingStep);

		// �ȼ� ��ġ���� ī�޶� ���� �������� �þ� ������ �����Ͽ��� �� 
		// ������ �����ϴ��� ���θ� �����ϰ�, ���� ������ �˾Ƴ���.
		const Range<float> RANGE =
			RayBoxIntersector::getValidRange(__Volume::__meta->volSize, STARTING_POINT, orthoBasis.u);

		// �þ� ������ ������ �������� �ʴ´ٸ�, �ȼ� ���� 0���� �����ϰ� �����Ѵ�.
		if (RANGE.end < RANGE.start)
		{
			pScreen[OFFSET].set(0.f);
			return;
		}

		// ���� ��ġ�� ���� ���� ���� �������� �̵��Ѵ�.
		Point3D samplePoint = (STARTING_POINT + (orthoBasis.u * RANGE.start));

		float transparency = 1.f; // ���� ����
		Color<float> color(0.f); // ���� RGB ����

		const Vector3D STEP_VECTOR = (orthoBasis.u * objectBasedSamplingStep);

		// __Volume indicator
		if (VolumeIndicator::recognize(
			__Volume::__meta->volSize, samplePoint, *__indicatorLength, *__indicatorThickness))
		{
			color += (*__indicatorColor * *__indicatorAlpha);
			transparency = (1.f - *__indicatorAlpha);
		}

		for (float t = RANGE.start; t < RANGE.end; t += objectBasedSamplingStep)
		{
			// ���� ���� ��ġ���� ���� ������ ���� ���� �����Ͽ� �����´�.
			// 0.5�� ���ϴ� ���� CUDA�� �ؽ�ó �޸� Ư�� ����. ������ �� ��

			const Point3D samplingPoint_mappedToMem = __mapIdxVolumeToMem(samplePoint);

			const float INTENSITY = tex3D(
				__Volume::__volTexture,
				samplingPoint_mappedToMem.x + .5f,
				samplingPoint_mappedToMem.y + .5f,
				samplingPoint_mappedToMem.z + .5f);

			// ���� ������ ���� �����Ѵ�.
			const float CORRECTED_INTENSITY = (INTENSITY * USHRT_MAX);

			// alpha transfer function�� ���� ���� �־� alpha ���� �˾Ƴ���.
			// ���� ����: ���� normalized �ε����� �ƴ�. ���� ���� ���� (RGB transfer function ��ο��� �ش��)
			const float ALPHA = tex1D(__transferTableAlpha, CORRECTED_INTENSITY + .5f);

			// �þ� ���� ���� ������ ���� alpha ���� �����Ѵ�.
			// (���� ������ �۾����� alpha ���� �ݹ� �����Ǿ� ���ϴ� ����� ���� �� ����.)
			const float CORRECTED_ALPHA = (1.f - powf((1.f - ALPHA), objectBasedSamplingStep));

			// alpha�� 0�� �ƴ� ��쿡 ���ؼ�
			if (!NumberUtility::nearEqual(ALPHA, 0.f))
			{
				// shading ������ �����Ѵ�.
				const Color<float> SHADING_RESULT = __shade(
					samplingPoint_mappedToMem, CORRECTED_INTENSITY, CORRECTED_ALPHA,
					shininess, pLightsDev, orthoBasis.u);

				// alpha-blending
				const float CUR_ALPHA = (transparency * CORRECTED_ALPHA);

				color.red += (CUR_ALPHA * SHADING_RESULT.red);
				color.green += (CUR_ALPHA * SHADING_RESULT.green);
				color.blue += (CUR_ALPHA * SHADING_RESULT.blue);

				transparency *= (1.f - CORRECTED_ALPHA);

				// ���� ������ 0.01f���� �۾����� 
				// �þ� ������ ������ �����Ѵ�. (early-ray termination)
				if (transparency < 0.01f)
					break;
			}

			// �þ� ������ �����Ѵ�.
			samplePoint += STEP_VECTOR;
		}

		// __Volume indicator
		if (transparency > .01f)
		{
			samplePoint = (STARTING_POINT + (orthoBasis.u * RANGE.end));
			if (VolumeIndicator::recognize(
				__Volume::__meta->volSize, samplePoint, *__indicatorLength, *__indicatorThickness))
				color += (*__indicatorColor * (transparency * *__indicatorAlpha));
		}

		// �ȼ� ���� alpha-blending compositing ���� ����� �����Ѵ�.
		pScreen[OFFSET].set(color.red, color.green, color.blue);
	}
}

////////////////////////////////
//// RENDERING ENGINE START ////
////////////////////////////////

/* constructor */
RenderingEngine::RenderingEngine() :
	volumeRenderer(__volumeMeta.numeric, __volumeLoaded), imageProcessor(__volumeMeta.numeric, __volumeLoaded)
{}

/* member function */
void RenderingEngine::loadVolume(const VolumeData& volumeData)
{
	__volumeMeta = volumeData.meta;

	__Device::__Volume::__free();
	__Device::__Volume::__malloc(volumeData);

	volumeRenderer.__init();
	imageProcessor.__init();

	__volumeLoaded = true;

	SystemIndirectAccessor::getEventBroadcaster().notifyVolumeLoaded(volumeData.meta);
}

bool RenderingEngine::isVolumeLoaded() const
{
	return __volumeLoaded;
}

const VolumeMeta &RenderingEngine::getVolumeMeta() const
{
	return __volumeMeta;
}

RenderingEngine& RenderingEngine::getInstance()
{
	return __instance;
}

///////////////////////////////
//// VOLUME RENDERER START ////
///////////////////////////////

/* constructor */
RenderingEngine::VolumeRenderer::VolumeRenderer(const VolumeNumericMeta &volumeNumericMeta, const bool &activeFlag) :
	__volumeNumericMeta(volumeNumericMeta), __volumeLoaded(activeFlag), camera(__imgBasedSamplingStep)
{
	cudaMalloc(&__Device::__pLightsDev, 3 * sizeof(Light));

	__Device::__transferTableRed.filterMode = cudaTextureFilterMode::cudaFilterModeLinear;
	__Device::__transferTableGreen.filterMode = cudaTextureFilterMode::cudaFilterModeLinear;
	__Device::__transferTableBlue.filterMode = cudaTextureFilterMode::cudaFilterModeLinear;
	__Device::__transferTableAlpha.filterMode = cudaTextureFilterMode::cudaFilterModeLinear;
}

/* member function */
void RenderingEngine::VolumeRenderer::render(
	Pixel *const pScreen, const int screenWidth, const int screenHeight)
{
	if (__volumeLoaded)
	{
		const OrthoBasis ORTHOBASIS = camera.getOrthoBasis();

		__Device::__raycast<<<dim3(screenWidth / 16, screenHeight / 16), dim3(16, 16)>>>(
			screenWidth, screenHeight, camera.getPosition(), ORTHOBASIS,
			__Device::__pLightsDev, __shininess, __imgBasedSamplingStep, __objectBasedSamplingStep, pScreen);
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

const Light &RenderingEngine::VolumeRenderer::getLight(const int index) const
{
	return __lights[index];
}

void RenderingEngine::VolumeRenderer::setIndicatorLength(const float length)
{
	cudaMemcpyToSymbol(__Device::__indicatorLength, &length, sizeof(float));
}

void RenderingEngine::VolumeRenderer::setIndicatorThickness(const float thickness)
{
	cudaMemcpyToSymbol(__Device::__indicatorThickness, &thickness, sizeof(float));
}

void RenderingEngine::VolumeRenderer::toggleLighting(const int index)
{
	__lights[index].enabled = !__lights[index].enabled;
	__syncLight();
}

void RenderingEngine::VolumeRenderer::setLightAmbient(const int index, const Color<float>& ambient)
{
	__lights[index].ambient = ambient;
	__syncLight();
}

void RenderingEngine::VolumeRenderer::setLightDiffuse(const int index, const Color<float>& diffuse)
{
	__lights[index].diffuse = diffuse;
	__syncLight();
}

void RenderingEngine::VolumeRenderer::setLightSpecular(const int index, const Color<float>& specular)
{
	__lights[index].specular = specular;
	__syncLight();
}

void RenderingEngine::VolumeRenderer::setLightXPos(const int index, const float x)
{
	__lights[index].position.x = x;
	__syncLight();
}

void RenderingEngine::VolumeRenderer::setLightYPos(const int index, const float y)
{
	__lights[index].position.y = y;
	__syncLight();
}

void RenderingEngine::VolumeRenderer::setLightZPos(const int index, const float z)
{
	__lights[index].position.z = z;
	__syncLight();
}

void RenderingEngine::VolumeRenderer::__init()
{
	__release();

	initLight();
	__initCamera();

	__pTransferFunction = new ColorTransferFunction(__volumeNumericMeta.voxelPrecision);

	const cudaChannelFormatDesc DESC = cudaCreateChannelDesc<float>();
	const size_t MEM_SIZE = (__volumeNumericMeta.voxelPrecision * sizeof(float));

	for (int i = 0; i < 4; i++)
		cudaMallocArray(__pTransferTableDevBufferArr + i, &DESC, MEM_SIZE);

	cudaBindTextureToArray(__Device::__transferTableRed, __pTransferTableDevBufferArr[0]);
	cudaBindTextureToArray(__Device::__transferTableGreen, __pTransferTableDevBufferArr[1]);
	cudaBindTextureToArray(__Device::__transferTableBlue, __pTransferTableDevBufferArr[2]);
	cudaBindTextureToArray(__Device::__transferTableAlpha, __pTransferTableDevBufferArr[3]);

	initTransferFunction(ColorChannelType::ALL);

	__initImgBasedSamplingStep();
	__objectBasedSamplingStep = 1.f;
}

bool RenderingEngine::VolumeRenderer::initTransferFunction(const ColorChannelType colorType)
{
	IF_F_RET_F(__pTransferFunction);

	__pTransferFunction->init(colorType);
	__syncTransferFunction(colorType);

	EventBroadcaster &eventBroadcaster = SystemIndirectAccessor::getEventBroadcaster();

	eventBroadcaster.notifyInitVolumeTransferFunction(colorType);
	eventBroadcaster.notifyUpdateVolumeTransferFunction(colorType);

	return true;
}

void RenderingEngine::VolumeRenderer::initLight()
{
	__lights[0].position = Constant::Light::Position::LEFT;
	__lights[0].ambient = { 0.03f, 0.02f, 0.01f };
	__lights[0].diffuse = { 0.4f, 0.1f, 0.1f };
	__lights[0].specular = { 0.5f, 0.1f, 0.1f };
	__lights[0].enabled = false;

	__lights[1].position = Constant::Light::Position::RIGHT;
	__lights[1].ambient = { 0.02f, 0.03f, 0.01f };
	__lights[1].diffuse = { 0.1f, 0.6f, 0.1f };
	__lights[1].specular = { 0.1f, 0.5f, 0.1f };
	__lights[1].enabled = false;

	__lights[2].position = Constant::Light::Position::BACK;
	__lights[2].ambient = { 0.01f, 0.02f, 0.03f };
	__lights[2].diffuse = { 0.2f, 0.4f, 0.6f };
	__lights[2].specular = { 0.3f, 0.4f, 0.5f };
	__lights[2].enabled = false;

	__syncLight();

	EventBroadcaster &eventBroadcaster = SystemIndirectAccessor::getEventBroadcaster();
	eventBroadcaster.notifyInitLight();
	eventBroadcaster.notifyUpdateLight();
}

void RenderingEngine::VolumeRenderer::__initCamera()
{
	camera.set(Constant::Camera::EYE, Constant::Camera::AT, Constant::Camera::UP);
}

void RenderingEngine::VolumeRenderer::__initImgBasedSamplingStep()
{
	__imgBasedSamplingStep =
		SQUARE(__volumeNumericMeta.volSize.width) +
		SQUARE(__volumeNumericMeta.volSize.height) +
		SQUARE(__volumeNumericMeta.volSize.depth);

	__imgBasedSamplingStep = (sqrtf(__imgBasedSamplingStep) / 400.f);
}

void RenderingEngine::VolumeRenderer::__release()
{
	if (__volumeLoaded)
	{
		cudaUnbindTexture(__Device::__transferTableRed);
		cudaUnbindTexture(__Device::__transferTableGreen);
		cudaUnbindTexture(__Device::__transferTableBlue);
		cudaUnbindTexture(__Device::__transferTableAlpha);

		delete __pTransferFunction;
		__pTransferFunction = nullptr;

		for (int i = 0; i < 4; i++)
		{
			cudaFreeArray(__pTransferTableDevBufferArr[i]);
			__pTransferTableDevBufferArr[i] = nullptr;
		}
	}
}

void RenderingEngine::VolumeRenderer::__syncLight()
{
	cudaMemcpy(
		__Device::__pLightsDev, __lights,
		3 * sizeof(Light), cudaMemcpyKind::cudaMemcpyHostToDevice);
}

void RenderingEngine::VolumeRenderer::__syncTransferFunction(const ColorChannelType colorType)
{
	const int MEM_SIZE = (__volumeNumericMeta.voxelPrecision * sizeof(float));

	if (colorType == ColorChannelType::ALL)
	{
		for (int i = 0; i < 4; i++)
		{
			cudaMemcpyToArray(
				__pTransferTableDevBufferArr[i], 0, 0,
				__pTransferFunction->getBuffer(static_cast<ColorChannelType>(i)),
				MEM_SIZE, cudaMemcpyKind::cudaMemcpyHostToDevice);
		}

		return;
	}

	cudaMemcpyToArray(
		__pTransferTableDevBufferArr[colorType], 0, 0,
		__pTransferFunction->getBuffer(colorType), MEM_SIZE, cudaMemcpyKind::cudaMemcpyHostToDevice);
}

RenderingEngine::VolumeRenderer::~VolumeRenderer()
{
	cudaFree(__Device::__pLightsDev);
}

///////////////////////////////
//// IMAGE PROCESSOR START ////
///////////////////////////////

namespace __Device
{
	namespace __ImgProc
	{
		texture<float> __transferTable;

		__device__
		Index2D<float> __calcSamplingPoint(
			const Index2D<> &screenIdx, const Size2D<> &screenSize,
			const float anchorHoriz, const float anchorVert, const float samplingStep)
		{
			const Size2D<float> SCR_SIZE_HALF = (screenSize.castTo<float>() / 2.f);

			const float SCR_HORIZ_OFFSET = ((static_cast<float>(screenIdx.x) - SCR_SIZE_HALF.width) * samplingStep);
			const float SCR_VERT_OFFSET = ((SCR_SIZE_HALF.height - static_cast<float>(screenIdx.y)) * samplingStep);

			Index2D<float> retVal = { anchorHoriz, anchorVert };

			retVal.x += SCR_HORIZ_OFFSET;
			retVal.y += SCR_VERT_OFFSET;

			return retVal;
		}

		__device__
		float __shade(const Point3D &samplingPoint)
		{
			const Point3D samplingPoint_mappedToMem = __Device::__mapIdxVolumeToMem(samplingPoint);

			const float INTENSITY =
				(tex3D(__Device::__Volume::__volTexture,
					samplingPoint_mappedToMem.x + .5f,
					samplingPoint_mappedToMem.y + .5f,
					samplingPoint_mappedToMem.z + .5f) *
					static_cast<float>(USHRT_MAX));

			return tex1D(__transferTable, INTENSITY + .5f);
		}

		__global__
		void __kernel_imgProcRender_top(
			Pixel* const pScreen, const Size2D<> screenSize, const Point2D anchor,
			const float samplingStep, const float volumeSlicingZ)
		{
			const Index2D<> SCR_IDX = Index2D<>::getKernelIndex();

			const int TEX_Y = ((screenSize.height - 1) - SCR_IDX.y);
			const int SCR_OFFSET = ((TEX_Y * screenSize.width) + SCR_IDX.x);

			const Index2D<float> SAMPLING_PLANE =
				__calcSamplingPoint(SCR_IDX, screenSize, anchor.x, anchor.y, samplingStep);

			const float PIXEL_VAL = __shade({ SAMPLING_PLANE.x, SAMPLING_PLANE.y, volumeSlicingZ });
			pScreen[SCR_OFFSET].set(PIXEL_VAL);
		}

		__global__
		void __kernel_imgProcRender_front(
			Pixel* const pScreen, const Size2D<> screenSize, const Point2D anchor,
			const float samplingStep, const float volumeSlicingY)
		{
			Index2D<> SCR_IDX = Index2D<>::getKernelIndex();
			SCR_IDX.y = ((screenSize.height - 1) - SCR_IDX.y);

			const int TEX_Y = ((screenSize.height - 1) - SCR_IDX.y);
			const int SCR_OFFSET = ((TEX_Y * screenSize.width) + SCR_IDX.x);

			const Index2D<float> SAMPLING_PLANE =
				__calcSamplingPoint(SCR_IDX, screenSize, anchor.x, anchor.y, samplingStep);

			const float PIXEL_VAL = __shade({ SAMPLING_PLANE.x, volumeSlicingY, SAMPLING_PLANE.y });
			pScreen[SCR_OFFSET].set(PIXEL_VAL);
		}

		__global__
		void __kernel_imgProcRender_right(
			Pixel* const pScreen, const Size2D<> screenSize, const Point2D anchor,
			const float samplingStep, const float volumeSlicingX)
		{
			using __Device::__Volume::__meta;

			const Index2D<> SCR_IDX = Index2D<>::getKernelIndex();

			const int TEX_Y = ((screenSize.height - 1) - SCR_IDX.y);
			const int SCR_OFFSET = ((TEX_Y * screenSize.width) + SCR_IDX.x);

			const Index2D<float> SAMPLING_PLANE =
				__calcSamplingPoint(SCR_IDX, screenSize, anchor.x, anchor.y, samplingStep);

			const float PIXEL_VAL = __shade({ volumeSlicingX, SAMPLING_PLANE.x, SAMPLING_PLANE.y });
			pScreen[SCR_OFFSET].set(PIXEL_VAL);
		}
	}
}

RenderingEngine::ImageProcessor::ImageProcessor(
	const VolumeNumericMeta &volumeNumericMeta, const bool &activeFlag) :
	__volumeNumericMeta(volumeNumericMeta), __volumeLoaded(activeFlag),
	__slicingPointMgr(__samplingStepArr), __anchorMgr(__samplingStepArr)
{
	using namespace __Device::__ImgProc;
	__transferTable.filterMode = cudaTextureFilterMode::cudaFilterModeLinear;
}

static float __calcSamplingStep(const float width, const float height)
{
	float retVal = (SQUARE(width) + SQUARE(height));

	return (sqrtf(retVal) / 400.f);
}

void RenderingEngine::ImageProcessor::__init()
{
	__release();

	__pTransferFunction = new MonoTransferFunction(__volumeNumericMeta.voxelPrecision);

	const cudaChannelFormatDesc DESC = cudaCreateChannelDesc<float>();
	cudaMallocArray(&__transferTableBuffer, &DESC, __volumeNumericMeta.voxelPrecision * sizeof(float));
	cudaBindTextureToArray(__Device::__ImgProc::__transferTable, __transferTableBuffer);

	initTransferFunction();

	const Size3D<float> &VOL_SIZE = __volumeNumericMeta.volSize;

	__slicingPointMgr.init(VOL_SIZE);
	__anchorMgr.init(VOL_SIZE);

	__samplingStepArr[SliceAxis::TOP] = __calcSamplingStep(VOL_SIZE.width, VOL_SIZE.height);
	__samplingStepArr[SliceAxis::FRONT] = __calcSamplingStep(VOL_SIZE.width, VOL_SIZE.depth);
	__samplingStepArr[SliceAxis::RIGHT] = __calcSamplingStep(VOL_SIZE.height, VOL_SIZE.depth);
}

void RenderingEngine::ImageProcessor::__release()
{
	if (__volumeLoaded)
	{
		cudaUnbindTexture(__Device::__ImgProc::__transferTable);

		cudaFreeArray(__transferTableBuffer);
		__transferTableBuffer = nullptr;

		delete __pTransferFunction;
		__pTransferFunction = nullptr;
	}
}

void RenderingEngine::ImageProcessor::__syncTransferFunction()
{
	cudaMemcpyToArray(
		__transferTableBuffer, 0, 0, __pTransferFunction->getBuffer(),
		__pTransferFunction->PRECISION * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);
}

Index2D<> RenderingEngine::ImageProcessor::
getSlicingPointForScreen(const Size2D<> &screenSize, const SliceAxis axis)
{
	const Point3D &slicingPointAdj = __slicingPointMgr.getSlicingPointAdj();

	return __anchorMgr.getSlicingPointForScreen(screenSize, slicingPointAdj, axis);
}

const Point3D &RenderingEngine::ImageProcessor::getSlicingPoint() const
{
	return __slicingPointMgr.getSlicingPoint();
}

void RenderingEngine::ImageProcessor::setSlicingPointFromScreen(
	const Size2D<> &screenSize, const Index2D<> &screenIdx, const SliceAxis axis)
{
	__slicingPointMgr.setSlicingPointFromScreen(
		screenSize, screenIdx, __anchorMgr.getAnchorAdj(axis), axis);
}

void RenderingEngine::ImageProcessor::setSlicingPointAdj(const Point3D &adj)
{
	__slicingPointMgr.setSlicingPointAdj(adj);
}

void RenderingEngine::ImageProcessor::setSlicingPoint(const Point3D &point)
{
	__slicingPointMgr.setSlicingPoint(point);
}

const Point2D &RenderingEngine::ImageProcessor::getAnchorAdj(const SliceAxis axis) const
{
	return __anchorMgr.getAnchorAdj(axis);
}

void RenderingEngine::ImageProcessor::setAnchorAdj(
	const float adjX, const float adjY, const SliceAxis axis)
{
	__anchorMgr.setAnchorAdj(adjX, adjY, axis);
}

void RenderingEngine::ImageProcessor::adjustSlicingPoint(const float delta, const SliceAxis axis)
{
	__slicingPointMgr.adjustSlicingPoint(delta, axis);
}

void RenderingEngine::ImageProcessor::adjustAnchor(
	const float deltaX, const float deltaY, const SliceAxis axis)
{
	__anchorMgr.adjustAnchor(deltaX, deltaY, axis);
}

void RenderingEngine::ImageProcessor::adjustSamplingStep(const float delta, const SliceAxis axis)
{
	__samplingStepArr[axis] =
		NumberUtility::truncate(__samplingStepArr[axis] + delta, .1f, 4.f);
}

bool RenderingEngine::ImageProcessor::initTransferFunction()
{
	IF_F_RET_F(__pTransferFunction);

	__pTransferFunction->init();
	__syncTransferFunction();

	EventBroadcaster &eventBroadcaster = SystemIndirectAccessor::getEventBroadcaster();
	eventBroadcaster.notifySliceTransferFunctionInit();
	eventBroadcaster.notifySliceTransferFunctionUpdate();

	return true;
}

void RenderingEngine::ImageProcessor::render(
	Pixel *const pScreen, const int screenWidth, const int screenHeight, const SliceAxis axis)
{
	if (__volumeLoaded)
	{
		using namespace __Device::__ImgProc;

		const dim3 gridDim =
		{
			static_cast<uint>(screenWidth / 16),
			static_cast<uint>(screenHeight / 16)
		};
		const dim3 blockDim = { 16, 16 };

		const Point3D &slicingPoint = __slicingPointMgr.getSlicingPoint();
		const Point2D &anchor = __anchorMgr.getAnchor(axis);

		switch (axis)
		{
		case SliceAxis::TOP:
			__kernel_imgProcRender_top<<<gridDim, blockDim>>>(
				pScreen, { screenWidth, screenHeight }, anchor, __samplingStepArr[axis], slicingPoint.z);
			break;

		case SliceAxis::FRONT:
			__kernel_imgProcRender_front<<<gridDim, blockDim>>>(
				pScreen, { screenWidth, screenHeight }, anchor, __samplingStepArr[axis], slicingPoint.y);
			break;

		case SliceAxis::RIGHT:
			__kernel_imgProcRender_right<<<gridDim, blockDim>>>(
				pScreen, { screenWidth, screenHeight }, anchor, __samplingStepArr[axis], slicingPoint.x);
			break;
		}
	}
}

RenderingEngine::ImageProcessor::~ImageProcessor()
{
	__release();
}