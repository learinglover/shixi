import { router, promptAction } from '@kit.ArkUI'
import featureAbility from '@ohos.ability.featureAbility'

interface TableRow {
  confidence: string
  personCount: string
  carCount: string
  targetCount: string
  duration: string
  path: string
  resultImage: Resource
}

interface SelectOption {
  value: string
  label: string
}

interface FilePickerResult {
  uri?: string;
  filePath?: string;
}

@Entry
@Component
struct Main {
  @State selectedTarget: string = '所有目标'
  @State tableData: TableRow[] = []
  @State targetOptions: SelectOption[] = [
    { value: '所有目标', label: '所有目标' },
    { value: '火焰', label: '火焰' },
    { value: '烟雾', label: '烟雾' },
    { value: '人员', label: '人员' }
  ]
  @State selectedImagePath: string = ''
  @State selectedVideoPath: string = ''
  @State showDrawer: boolean = false

  showTodo(msg: string): void {
    promptAction.showToast({ message: msg, duration: 1500 })
  }

  async runProcess(): Promise<void> {
    this.showTodo('正在运行...');
    const newItem: TableRow = { confidence: '0.95', personCount: (this.tableData.length + 1).toString(), carCount: (this.tableData.length + 1).toString(), targetCount: (this.tableData.length + 1).toString(), duration: '0.5', path: '', resultImage: $r('app.media.fire') };
    this.tableData.push(newItem);
  }

  async selectImageFile(): Promise<void> {
    this.showTodo('Stage模型下请通过自定义ExtensionAbility实现文件选择功能');
    await this.runProcess();
  }

  async selectVideoFile(): Promise<void> {
    this.showTodo('Stage模型下请通过自定义ExtensionAbility实现视频选择功能');
    await this.runProcess();
  }

  build() {
    Stack() {
      // 主内容区
      Column() {
        // 顶部标题栏（包含菜单按钮）
        Row() {
          Button() {
            Image($r('app.media.folder'))
              .width(24)
              .height(24)
          }
          .width(40)
          .height(40)
          .backgroundColor('transparent')
          .onClick(() => { this.showDrawer = true })
          Text('人车数量统计系统')
            .fontSize(20)
            .fontWeight(FontWeight.Bold)
            .fontColor('#7C4DFF')
            .margin({ left: 8 })
        }
        .width('100%')
        .height(56)
        .padding({ left: 16, right: 16 })
        .backgroundColor('#F3E5F5')
        .alignItems(VerticalAlign.Center)

        // 操作按钮区
        Row() {
          Button('选择图片')
            .width(90)
            .height(36)
            .fontSize(14)
            .backgroundColor('#7C4DFF')
            .fontColor('#fff')
            .borderRadius(18)
            .margin({ right: 8 })
            .onClick(() => this.selectImageFile())
          Button('选择视频')
            .width(90)
            .height(36)
            .fontSize(14)
            .backgroundColor('#9575CD')
            .fontColor('#fff')
            .borderRadius(18)
            .margin({ right: 8 })
            .onClick(() => this.selectVideoFile())
          Button('摄像头')
            .width(90)
            .height(36)
            .fontSize(14)
            .backgroundColor('#B39DDB')
            .fontColor('#fff')
            .borderRadius(18)
            .margin({ right: 8 })
            .onClick(() => { this.showTodo('摄像头功能开发中'); this.runProcess(); })
          Button('运行')
            .width(90)
            .height(36)
            .fontSize(14)
            .backgroundColor('#4CAF50')
            .fontColor('#fff')
            .borderRadius(18)
            .onClick(() => this.runProcess())
        }
        .margin({ top: 16, bottom: 14 })
        .alignItems(VerticalAlign.Center)
        .justifyContent(FlexAlign.Center)

        // 数据表格
        Column() {
          // 表头
          Row() {
            Text('序号').width(40).fontWeight(FontWeight.Bold).fontColor('#7C4DFF').fontSize(14).textAlign(TextAlign.Center)
            Text('置信度').width(80).fontWeight(FontWeight.Bold).fontColor('#7C4DFF').fontSize(14).textAlign(TextAlign.Center)
            Text('人员数量').width(80).fontWeight(FontWeight.Bold).fontColor('#7C4DFF').fontSize(14).textAlign(TextAlign.Center)
            Text('车辆数量').width(80).fontWeight(FontWeight.Bold).fontColor('#7C4DFF').fontSize(14).textAlign(TextAlign.Center)
          }
          .height(28)
          .backgroundColor('#F3E5F5')
          .borderRadius(8)
          .margin({ bottom: 2 })
          ForEach(this.tableData, (item: TableRow, index: number) => {
            Row() {
              Text((index+1).toString()).width(40).fontSize(13).textAlign(TextAlign.Center)
              Text(item.confidence).width(80).fontSize(13).textAlign(TextAlign.Center)
              Text(item.personCount).width(80).fontSize(13).textAlign(TextAlign.Center)
              Text(item.carCount).width(80).fontSize(13).textAlign(TextAlign.Center)
            }
            .height(26)
            .backgroundColor(index % 2 === 0 ? '#fff' : '#F8F6FC')
            .borderRadius(6)
          })
        }
        .width('96%')
        .margin({ top: 10 })
        .backgroundColor('#F8F6FC')
        .borderRadius(12)
        .padding({ left: 6, right: 6, top: 6, bottom: 6 })
        .alignSelf(ItemAlign.Center)
        .layoutWeight(1)

        // 运行结果卡片
        Column() {
          Text('运行结果')
            .fontSize(16)
            .fontWeight(FontWeight.Bold)
            .fontColor('#7C4DFF')
            .margin({ bottom: 6 })
          Text('运行过程已结束，结果已更新至表格。')
            .fontSize(14)
            .fontColor('#333')
          if (this.tableData.length > 0) {
            Swiper() {
              ForEach(this.tableData, (item: TableRow) => {
                Image(item.resultImage)
                  .width('100%')
                  .height(200)
                  .objectFit(ImageFit.Cover)
                  .margin({ top: 10, bottom: 10 })
              })
            }
            .width('100%')
            .height(220)
            .loop(true)
            .indicator(true)
            .margin({ top: 10, bottom: 10 })
          }
        }
        .width('96%')
        .margin({ top: 10, bottom: 10 })
        .backgroundColor('#F8F6FC')
        .borderRadius(12)
        .padding({ top: 10, bottom: 10, left: 10, right: 10 })
        .alignSelf(ItemAlign.Center)
      }
      .width('100%')
      .height('100%')
      .backgroundColor('#F3E5F5')

      // 侧边栏和遮罩层
      if (this.showDrawer) {
        // 侧边栏
        Column() {
          Text('菜单')
            .fontSize(20)
            .fontWeight(FontWeight.Bold)
            .fontColor('#7C4DFF')
            .margin({ top: 18, bottom: 20 })
            .alignSelf(ItemAlign.Center)
          Button('历史运行记录')
            .width('90%')
            .height(40)
            .fontSize(16)
            .backgroundColor('#7C4DFF')
            .fontColor('#fff')
            .borderRadius(20)
            .margin({ bottom: 12 })
            .onClick(() => { this.showDrawer = false; this.showTodo('查看历史运行记录') })
          Button('常用')
            .width('90%')
            .height(40)
            .fontSize(16)
            .backgroundColor('#9575CD')
            .fontColor('#fff')
            .borderRadius(20)
            .margin({ bottom: 12 })
            .onClick(() => { this.showDrawer = false; this.showTodo('常用功能') })
        }
        .backgroundColor('#F3E5F5')
        .width(220)
        .height('100%')
        .padding({ left: 16, right: 16 })
        .position({ left: 0, top: 0 })

        // 遮罩层
        Row() {}
        .backgroundColor('rgba(0,0,0,0.2)')
        .onClick(() => { this.showDrawer = false })
        .width('100%')
        .height('100%')
        .position({ left: 220, top: 0 })
      }
    }
    .width('100%').height('100%')
  }
} 