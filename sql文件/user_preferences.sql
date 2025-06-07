/*
 Navicat Premium Data Transfer

 Source Server         : local
 Source Server Type    : MySQL
 Source Server Version : 80036 (8.0.36)
 Source Host           : localhost:3306
 Source Schema         : software

 Target Server Type    : MySQL
 Target Server Version : 80036 (8.0.36)
 File Encoding         : 65001

 Date: 04/06/2025 20:41:38
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for user_preferences
-- ----------------------------
DROP TABLE IF EXISTS `user_preferences`;
CREATE TABLE `user_preferences`  (
  `id` int NOT NULL AUTO_INCREMENT,
  `user_id` int NOT NULL COMMENT '关联的用户ID',
  `common_commands` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '常用指令 (JSON格式)',
  `interaction_habits` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '交互习惯 (JSON格式)',
  `command_aliases` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '指令别名 (JSON格式)',
  `updated_at` datetime NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '最后更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `user_id_unique`(`user_id` ASC) USING BTREE COMMENT '确保每个用户只有一条偏好设置记录',
  INDEX `ix_user_preferences_id`(`id` ASC) USING BTREE,
  CONSTRAINT `fk_user_preferences_user_id` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE = InnoDB AUTO_INCREMENT = 2 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = '用户个性化偏好设置表' ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of user_preferences
-- ----------------------------
INSERT INTO `user_preferences` VALUES (1, 6, '{\r\n    \"导航到公司\": 20,\r\n    \"播放Melody\": 10,\r\n    \"打开空调\": 5,\r\n    \"查看天气\": 2,\r\n    \"关闭车窗\": 1\r\n  }', '{\r\n    \"早高峰使用\": {\r\n      \"频率\": \"高\",\r\n      \"时间段\": \"07:00-09:00\"\r\n    },\r\n    \"晚高峰使用\": {\r\n      \"频率\": \"中\",\r\n      \"时间段\": \"17:00-19:00\"\r\n    }\r\n  }', '{\r\n    \"导航到公司\": \"去公司\",\r\n    \"播放Melody\": \"放Melody\",\r\n    \"打开空调\": \"开空调\"\r\n  }', '2025-06-04 13:29:12');

SET FOREIGN_KEY_CHECKS = 1;
